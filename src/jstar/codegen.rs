//! x86-64 Code Generation — Phase 5 of the JStar compiler.
//!
//! Direct x86-64 machine code emission. No LLVM. No Cranelift.
//! The bootstrap compiler must have zero external codegen dependencies
//! so the self-hosted JStar compiler inherits none.
//!
//! Calling convention: System V AMD64 ABI
//!   Args: rdi, rsi, rdx, rcx, r8, r9 (integers/pointers)
//!   Return: rax
//!   Callee-saved: rbx, rbp, r12-r15
//!   Caller-saved: rax, rcx, rdx, rsi, rdi, r8-r11
//!
//! Register allocation: simple linear mapping from virtual registers.
//! SSA form from IR makes this straightforward.

use super::grammar::JStarType;
use super::ir::*;
use crate::types::{MorphResult, MorphlexError};

// ─── x86-64 Register Encoding ───────────────────────────────────────────────

/// x86-64 general-purpose registers (64-bit names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum X86Reg {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl X86Reg {
    /// The 3-bit register encoding for ModR/M and SIB bytes.
    fn encoding(self) -> u8 {
        (self as u8) & 0x07
    }

    /// Whether this register requires a REX.B or REX.R prefix (r8-r15).
    fn needs_rex_ext(self) -> bool {
        (self as u8) >= 8
    }
}

#[allow(dead_code)]
const SCRATCH_REGS: [X86Reg; 7] = [
    X86Reg::Rax,
    X86Reg::Rcx,
    X86Reg::Rdx,
    X86Reg::Rsi,
    X86Reg::Rdi,
    X86Reg::R8,
    X86Reg::R9,
];

// ─── Machine Code Buffer ────────────────────────────────────────────────────

/// The output of code generation: raw x86-64 machine code bytes.
#[derive(Debug, Clone)]
pub struct MachineCode {
    /// The .text section (executable code)
    pub text: Vec<u8>,
    /// The .data section (initialized data — string literals only)
    pub data: Vec<u8>,
    /// Size of uninitialized global data (BSS). This region follows .data in
    /// virtual memory but is NOT stored in the file — the kernel zero-fills it.
    pub bss_size: usize,
    /// Stack frame size for _start
    pub stack_size: usize,
    /// Virtual address of .data section (set by linker, stored for codegen)
    pub data_vaddr: u64,
    /// Positions in .text where data-section offsets need patching by the linker.
    /// Each entry is the byte offset in .text of an 8-byte value to which the
    /// linker adds data_vaddr.
    pub data_fixups: Vec<usize>,
}

/// Generate x86-64 machine code from IR.
pub fn generate(program: &IrProgram) -> MorphResult<MachineCode> {
    let mut emitter = CodeGen::new();

    // .data section: only string literals (initialized data that must be in the file).
    emitter.data = program.string_data.clone();

    // BSS: uninitialized global data. Lives after .data in virtual memory but
    // is NOT written to the file — the kernel zero-fills it via p_memsz > p_filesz.
    let global_base = emitter.data.len();
    emitter.bss_size = program.global_data.len();

    // Build the global_vregs map with adjusted offsets
    for (&vreg, &offset) in &program.global_vregs {
        emitter.global_vregs.insert(vreg, global_base + offset);
        let is_direct_storage = program
            .global_vars
            .values()
            .find(|(var_offset, _, _)| *var_offset == offset)
            .map(|(_, alloc_size, ty)| *alloc_size > ty.size_bytes())
            .unwrap_or(false);
        if is_direct_storage {
            emitter.direct_storage_vregs.insert(vreg);
        }
    }

    for func in &program.functions {
        emitter.emit_function(func)?;
    }

    // Resolve call fixups now that all function offsets are known
    emitter.apply_call_fixups()?;

    Ok(MachineCode {
        text: emitter.text,
        data: emitter.data,
        bss_size: emitter.bss_size,
        stack_size: emitter.stack_size,
        data_vaddr: 0, // set by linker
        data_fixups: emitter.data_fixups,
    })
}

struct CodeGen {
    text: Vec<u8>,
    data: Vec<u8>,
    bss_size: usize,
    stack_size: usize,
    /// Map virtual register -> stack offset from rbp
    vreg_offsets: std::collections::HashMap<VReg, i32>,
    next_stack_offset: i32,
    /// Label name -> byte offset in .text (recorded when emitting each block)
    label_offsets: std::collections::HashMap<String, usize>,
    /// (patch_offset, target_label) — forward references to be resolved after emission
    fixups: Vec<(usize, String)>,
    /// Function name -> byte offset in .text (for call resolution)
    function_offsets: std::collections::HashMap<String, usize>,
    /// (patch_offset, function_name) — call fixups
    call_fixups: Vec<(usize, String)>,
    /// Whether current function is _start (uses sys_exit) vs regular (uses ret)
    is_entry_point: bool,
    /// Global vregs: vreg -> data_offset (absolute offset in .data section).
    /// If a vreg is in this map, it lives in .data, not on the stack.
    global_vregs: std::collections::HashMap<VReg, usize>,
    /// Vregs whose identity is the storage object itself, not a slot holding a value.
    ///
    /// Examples:
    /// - `a byte buf 256` -> `buf` is inline storage; indexed access should use its address
    /// - `allocate 256` returns a pointer value; indexed access must load that pointer first
    /// - `addressof buf` returns a pointer value; same rule
    direct_storage_vregs: std::collections::HashSet<VReg>,
    /// Positions in .text where data-section offsets need linker patching.
    data_fixups: Vec<usize>,
}

impl CodeGen {
    fn new() -> Self {
        CodeGen {
            text: Vec::new(),
            data: Vec::new(),
            bss_size: 0,
            stack_size: 0,
            vreg_offsets: std::collections::HashMap::new(),
            next_stack_offset: -8, // first slot at rbp-8
            label_offsets: std::collections::HashMap::new(),
            fixups: Vec::new(),
            function_offsets: std::collections::HashMap::new(),
            call_fixups: Vec::new(),
            is_entry_point: false,
            global_vregs: std::collections::HashMap::new(),
            direct_storage_vregs: std::collections::HashSet::new(),
            data_fixups: Vec::new(),
        }
    }

    /// Allocate a stack slot for a virtual register. Returns offset from rbp.
    fn alloc_stack_slot(&mut self, vreg: VReg, size: usize) -> i32 {
        let aligned_size = ((size + 7) / 8) * 8; // 8-byte align
        self.next_stack_offset -= aligned_size as i32;
        let offset = self.next_stack_offset;
        self.vreg_offsets.insert(vreg, offset);
        offset
    }

    /// Get the stack offset for a virtual register.
    /// Panics with a diagnostic if the vreg was never allocated a stack slot.
    fn vreg_offset(&self, vreg: VReg) -> i32 {
        *self.vreg_offsets.get(&vreg).unwrap_or_else(|| {
            panic!(
                "codegen: vreg {} has no stack slot (allocated vregs: {:?})",
                vreg,
                self.vreg_offsets.keys().collect::<Vec<_>>()
            )
        })
    }

    fn emit_function(&mut self, func: &IrFunction) -> MorphResult<()> {
        // Reset per-function state
        self.vreg_offsets.clear();
        self.next_stack_offset = -8;
        self.label_offsets.clear();
        self.fixups.clear();
        self.is_entry_point = func.name == "_start";

        // Record function offset for call resolution
        self.function_offsets.insert(func.name.clone(), self.text.len());

        // === CRITICAL FIX BLOCK - MUST BE FIRST ===
        // Apply ALL data fixups BEFORE any code is emitted
        let data_start = self.data.len();
        for fixup_offset in &self.data_fixups {
            if *fixup_offset + 8 <= self.text.len() {
                let offset_ptr = &mut self.text[*fixup_offset..*fixup_offset + 8];
                let current = u64::from_le_bytes(offset_ptr.try_into().unwrap());
                let fixed = current + data_start as u64;
                offset_ptr.copy_from_slice(&fixed.to_le_bytes());
            }
        }

        // === END CRITICAL FIX BLOCK ===

        // Restore direct_storage_vregs for this function
        self.direct_storage_vregs
            .retain(|vreg| self.global_vregs.contains_key(vreg));

        // Pre-allocate stack slots for all virtual registers
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    IrInst::Alloca { dest, size, .. } => {
                        if self.instruction_allocates_direct_storage(inst) {
                            self.direct_storage_vregs.insert(*dest);
                        }
                        self.alloc_stack_slot(*dest, *size);
                    }
                    IrInst::BinOp { dest, ty, .. }
                    | IrInst::UnaryOp { dest, ty, .. }
                    | IrInst::Copy { dest, ty, .. }
                    | IrInst::Load { dest, ty, .. } => {
                        self.alloc_stack_slot(*dest, ty.size_bytes().max(8));
                    }
                    IrInst::Compare { dest, .. }
                    | IrInst::Call { dest, .. }
                    | IrInst::Syscall { dest, .. }
                    | IrInst::AddressOf { dest, .. } => {
                        self.alloc_stack_slot(*dest, 8);
                    }
                    IrInst::LoadIndexed { dest, .. } => {
                        self.alloc_stack_slot(*dest, 8);
                    }
                    IrInst::ArrayAlloc { dest, count } => {
                        self.alloc_stack_slot(*dest, *count * 8);
                    }
                    IrInst::ArrayLoad { dest, .. }
                    | IrInst::ArrayLength { dest, .. }
                    | IrInst::HashOp { dest, .. }
                    | IrInst::FileOpen { dest, .. }
                    | IrInst::FileRead { dest, .. }
                    | IrInst::StrCmp { dest, .. }
                    | IrInst::StrLen { dest, .. } => {
                        self.alloc_stack_slot(*dest, 8);
                    }
                    // Kernel instructions with dest register
                    IrInst::Inb { dest, .. }
                    | IrInst::Rdmsr { dest, .. }
                    | IrInst::ReadCr { dest, .. }
                    | IrInst::LoadAbs { dest, .. } => {
                        self.alloc_stack_slot(*dest, 8);
                    }
                    // Kernel instructions without dest
                    IrInst::Outb { .. } | IrInst::Cli | IrInst::Sti
                    | IrInst::Lgdt { .. } | IrInst::Lidt { .. }
                    | IrInst::Wrmsr { .. } | IrInst::Invlpg { .. }
                    | IrInst::WriteCr { .. } | IrInst::Iretq
                    | IrInst::Ltr { .. } | IrInst::StoreAbs { .. } => {}
                    IrInst::Store { .. } | IrInst::StoreIndexed { .. }
                    | IrInst::Print { .. } | IrInst::PrintStr { .. }
                    | IrInst::Nop | IrInst::ArrayStore { .. }
                    | IrInst::FileClose { .. }
                    | IrInst::StrCopy { .. } => {}
                }
            }
        }

        // Calculate total stack frame size (16-byte aligned per ABI)
        self.stack_size = ((-self.next_stack_offset) as usize + 15) & !15;

        // Function prologue: push rbp; mov rbp, rsp
        self.emit_push_reg(X86Reg::Rbp);
        self.emit_mov_reg_reg(X86Reg::Rbp, X86Reg::Rsp);
        
        // Stack probing for large frames (>4KB)
        if self.stack_size > 0 {
            if self.stack_size > 4096 {
                let mut remaining = self.stack_size;
                while remaining > 4096 {
                    self.emit_sub_reg_imm(X86Reg::Rsp, 4096);
                    self.text.extend_from_slice(&[0x48, 0x8B, 0x04, 0x24]); // touch [rsp]
                    remaining -= 4096;
                }
                if remaining > 0 {
                    self.emit_sub_reg_imm(X86Reg::Rsp, remaining as i32);
                }
            } else {
                self.emit_sub_reg_imm(X86Reg::Rsp, self.stack_size as i32);
            }
        }

        // For non-_start functions: store incoming arguments from registers to stack.
        // Only the first param_count Alloca instructions are parameter slots;
        // subsequent Allocas are local variables and must not receive ABI args.
        if !self.is_entry_point && func.param_count > 0 {
            let arg_regs = [
                X86Reg::Rdi,
                X86Reg::Rsi,
                X86Reg::Rdx,
                X86Reg::Rcx,
                X86Reg::R8,
                X86Reg::R9,
            ];
            let mut param_vregs: Vec<VReg> = Vec::new();
            if let Some(entry_block) = func.blocks.first() {
                for inst in &entry_block.instructions {
                    if let IrInst::Alloca { dest, .. } = inst {
                        if param_vregs.len() < func.param_count {
                            param_vregs.push(*dest);
                        }
                    }
                }
            }
            if param_vregs.len() != func.param_count {
                return Err(MorphlexError::CodegenError(format!(
                    "function '{}': expected {} parameter allocas in entry block, found {}",
                    func.name, func.param_count, param_vregs.len()
                )));
            }
            for (i, vreg) in param_vregs.iter().enumerate() {
                if i >= arg_regs.len() {
                    break;
                }
                let offset = self.vreg_offset(*vreg);
                self.emit_store_reg_to_rbp_offset(arg_regs[i], offset);
            }
        }

        // Emit each basic block, recording label offsets
        for block in &func.blocks {
            self.label_offsets
                .insert(block.label.clone(), self.text.len());
            self.emit_block(block)?;
        }

        // Resolve all fixups (forward jumps patched with actual offsets)
        self.apply_fixups()?;

        Ok(())
    }

    fn emit_block(&mut self, block: &BasicBlock) -> MorphResult<()> {
        for inst in &block.instructions {
            self.emit_instruction(inst)?;
        }
        self.emit_terminator(&block.terminator)?;
        Ok(())
    }

    fn emit_instruction(&mut self, inst: &IrInst) -> MorphResult<()> {
        match inst {
            IrInst::BinOp {
                dest, op, lhs, rhs, ..
            } => {
                // Load lhs into rax
                self.emit_load_value(X86Reg::Rax, lhs);
                // Load rhs into rcx
                self.emit_load_value(X86Reg::Rcx, rhs);

                match op {
                    IrBinOp::Add => self.emit_add_reg_reg(X86Reg::Rax, X86Reg::Rcx),
                    IrBinOp::Sub => self.emit_sub_reg_reg(X86Reg::Rax, X86Reg::Rcx),
                    IrBinOp::Mul => self.emit_imul_reg_reg(X86Reg::Rax, X86Reg::Rcx),
                    IrBinOp::Div => {
                        // idiv: rdx:rax / rcx -> rax (quotient), rdx (remainder)
                        self.emit_cqo(); // sign-extend rax into rdx
                        self.emit_idiv(X86Reg::Rcx);
                    }
                    IrBinOp::Mod => {
                        self.emit_cqo();
                        self.emit_idiv(X86Reg::Rcx);
                        // Remainder is in rdx, move to rax
                        self.emit_mov_reg_reg(X86Reg::Rax, X86Reg::Rdx);
                    }
                    IrBinOp::And => self.emit_and_reg_reg(X86Reg::Rax, X86Reg::Rcx),
                    IrBinOp::Or => self.emit_or_reg_reg(X86Reg::Rax, X86Reg::Rcx),
                    IrBinOp::Xor => self.emit_xor_reg_reg(X86Reg::Rax, X86Reg::Rcx),
                    IrBinOp::Shl => self.emit_shl_reg_cl(X86Reg::Rax),
                    IrBinOp::Shr => self.emit_shr_reg_cl(X86Reg::Rax),
                }

                // Store result from rax to dest's stack slot
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::UnaryOp { dest, op, src, .. } => {
                self.emit_load_value(X86Reg::Rax, src);
                match op {
                    IrUnaryOp::Neg => self.emit_neg(X86Reg::Rax),
                    IrUnaryOp::Not => self.emit_not(X86Reg::Rax),
                }
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Copy { dest, src, .. } => {
                self.emit_load_value(X86Reg::Rax, src);
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Compare {
                dest,
                lhs,
                rhs,
                kind,
                ..
            } => {
                self.emit_load_value(X86Reg::Rax, lhs);
                self.emit_load_value(X86Reg::Rcx, rhs);
                self.emit_cmp_reg_reg(X86Reg::Rax, X86Reg::Rcx);
                // Set result based on CmpKind (setcc al; movzx rax, al)
                self.emit_setcc(*kind);
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Load { dest, addr, ty } => {
                match addr {
                    IrValue::Reg(vreg) if self.is_global_vreg(*vreg) => {
                        // Global variable: load from .data section
                        let data_offset = self.global_vregs[vreg];
                        if Self::is_byte_type(ty) {
                            self.emit_load_byte_global(data_offset);
                        } else {
                            self.emit_load_global(data_offset);
                        }
                    }
                    // Stack-slot variable: load directly from [rbp+offset]
                    IrValue::Reg(vreg) => {
                        let src_offset = self.vreg_offset(*vreg);
                        if Self::is_byte_type(ty) {
                            self.emit_load_byte_from_rbp_offset(X86Reg::Rax, src_offset);
                        } else {
                            self.emit_load_from_rbp_offset(X86Reg::Rax, src_offset);
                        }
                    }
                    // Pointer/address: dereference via [rax]
                    _ => {
                        self.emit_load_value(X86Reg::Rax, addr);
                        self.emit_load_indirect(X86Reg::Rax, X86Reg::Rax);
                    }
                }
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Store { addr, value, ty } => {
                self.emit_load_value(X86Reg::Rcx, value);
                match addr {
                    IrValue::Reg(vreg) if self.is_global_vreg(*vreg) => {
                        // Global variable: store to .data section
                        let data_offset = self.global_vregs[vreg];
                        if Self::is_byte_type(ty) {
                            self.emit_store_byte_global(data_offset);
                        } else {
                            self.emit_store_global(data_offset);
                        }
                    }
                    // Stack-slot variable: store directly to [rbp+offset]
                    IrValue::Reg(vreg) => {
                        let offset = self.vreg_offset(*vreg);
                        if Self::is_byte_type(ty) {
                            self.emit_store_byte_to_rbp_offset(X86Reg::Rcx, offset);
                        } else {
                            self.emit_store_reg_to_rbp_offset(X86Reg::Rcx, offset);
                        }
                    }
                    // Pointer/address: store via [rax]
                    _ => {
                        self.emit_load_value(X86Reg::Rax, addr);
                        self.emit_store_indirect(X86Reg::Rax, X86Reg::Rcx);
                    }
                }
            }

            IrInst::Syscall { dest, number, args } => {
                // Linux syscall ABI: rax=number, rdi/rsi/rdx/r10/r8/r9=args
                self.emit_load_value(X86Reg::Rax, number);
                let arg_regs = [
                    X86Reg::Rdi,
                    X86Reg::Rsi,
                    X86Reg::Rdx,
                    X86Reg::R10,
                    X86Reg::R8,
                    X86Reg::R9,
                ];
                for (i, arg) in args.iter().enumerate() {
                    if i < arg_regs.len() {
                        self.emit_load_value(arg_regs[i], arg);
                    }
                }
                self.emit_syscall();
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Call {
                dest, name, args, ..
            } => {
                // Load arguments into System V ABI registers
                let arg_regs = [
                    X86Reg::Rdi,
                    X86Reg::Rsi,
                    X86Reg::Rdx,
                    X86Reg::Rcx,
                    X86Reg::R8,
                    X86Reg::R9,
                ];
                for (i, arg) in args.iter().enumerate() {
                    if i < arg_regs.len() {
                        self.emit_load_value(arg_regs[i], arg);
                    }
                }
                // Emit call rel32 (0xE8 + 4-byte signed displacement)
                self.text.push(0xE8);
                let patch_offset = self.text.len();
                self.text.extend_from_slice(&0i32.to_le_bytes()); // placeholder
                self.call_fixups.push((patch_offset, name.clone()));
                // Store return value (rax) to dest's stack slot
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Print { value } => {
                self.emit_print_integer(value);
            }

            IrInst::PrintStr { data_offset, len } => {
                self.emit_print_string(*data_offset, *len);
            }

            IrInst::StoreIndexed {
                base,
                index,
                value,
                ty,
            } => {
                self.emit_index_base(X86Reg::Rax, *base);
                // Load index into rcx
                self.emit_load_value(X86Reg::Rcx, index);
                // Load value into rdx
                self.emit_load_value(X86Reg::Rdx, value);
                // Store: [rax + rcx] = rdx (byte or qword)
                if Self::is_byte_type(ty) {
                    // mov byte [rax+rcx], dl
                    // REX prefix (0x40 for byte reg access)
                    self.text.push(0x40);
                    self.text.push(0x88); // MOV r/m8, r8
                    // ModR/M: [rax+rcx] with SIB = mod 00, reg=rdx(2), rm=100(SIB)
                    self.text.push(0x14); // 00 010 100
                    // SIB: scale=00(1), index=rcx(001), base=rax(000)
                    self.text.push(0x08); // 00 001 000
                } else {
                    // mov qword [rax+rcx*8], rdx
                    self.text.push(0x48); // REX.W
                    self.text.push(0x89); // MOV r/m64, r64
                    self.text.push(0x14); // ModR/M: [SIB], reg=rdx(2)
                    // SIB: scale=11(8), index=rcx(001), base=rax(000)
                    self.text.push(0xC8); // 11 001 000
                }
            }

            IrInst::LoadIndexed {
                dest,
                base,
                index,
                ty,
            } => {
                self.emit_index_base(X86Reg::Rax, *base);
                // Load index into rcx
                self.emit_load_value(X86Reg::Rcx, index);
                if Self::is_byte_type(ty) {
                    // movzx rax, byte [rax+rcx]
                    self.text.push(0x48); // REX.W
                    self.text.push(0x0F);
                    self.text.push(0xB6); // MOVZX r64, r/m8
                    // ModR/M: [SIB], reg=rax(0)
                    self.text.push(0x04); // 00 000 100
                    // SIB: scale=00(1), index=rcx(001), base=rax(000)
                    self.text.push(0x08); // 00 001 000
                } else {
                    // mov rax, [rax+rcx*8]
                    self.text.push(0x48); // REX.W
                    self.text.push(0x8B); // MOV r64, r/m64
                    self.text.push(0x04); // ModR/M: [SIB], reg=rax(0)
                    // SIB: scale=11(8), index=rcx(001), base=rax(000)
                    self.text.push(0xC8); // 11 001 000
                }
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Alloca { .. } => {
                // Stack space already allocated in prologue
            }

            IrInst::AddressOf { dest, src } => {
                if self.is_global_vreg(*src) {
                    // Global: the address IS the .data section address
                    let data_offset = self.global_vregs[src];
                    self.emit_load_global_addr(data_offset);
                } else {
                    // Local: lea rax, [rbp+src_offset]
                    let src_offset = self.vreg_offset(*src);
                    self.emit_lea_rbp_offset(X86Reg::Rax, src_offset);
                }
                let dest_offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, dest_offset);
            }

            IrInst::ArrayAlloc { .. } => {
                // Stack space already allocated in prologue
            }

            IrInst::ArrayLoad { dest, base, index } => {
                self.emit_load_indexable_base(X86Reg::Rax, base);
                // Load index into rcx
                self.emit_load_value(X86Reg::Rcx, index);
                // rax = *(rax + rcx * 8): mov rax, [rax + rcx*8]
                // SIB: base=rax(0), index=rcx(1), scale=8(3)
                // REX.W + mov rax, [rax + rcx*8]
                self.text.extend_from_slice(&[0x48, 0x8B, 0x04, 0xC8]);
                let dest_offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, dest_offset);
            }

            IrInst::ArrayStore { base, index, value } => {
                // Load value into rdx
                self.emit_load_value(X86Reg::Rdx, value);
                // Load base address into rax
                self.emit_load_indexable_base(X86Reg::Rax, base);
                // Load index into rcx
                self.emit_load_value(X86Reg::Rcx, index);
                // mov [rax + rcx*8], rdx
                // REX.W + mov [rax + rcx*8], rdx
                self.text.extend_from_slice(&[0x48, 0x89, 0x14, 0xC8]);
                // ^ 0x89 = MOV r/m64, r64; ModRM=0x14 (SIB, no disp); SIB=0xC8 (rcx*8+rax)
            }

            IrInst::ArrayLength { dest, count } => {
                self.emit_mov_reg_imm64(X86Reg::Rax, *count as i64);
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::HashOp { dest, addr, len } => {
                // FNV-1a hash: compact inline loop
                // rsi = base addr, rcx = len, result in eax
                self.emit_load_indexable_base(X86Reg::Rsi, addr);
                self.emit_load_value(X86Reg::Rcx, len); // byte count

                // Initialize hash = FNV offset basis (2166136261 = 0x811C9DC5)
                self.emit_mov_reg_imm64(X86Reg::Rax, 0x811C9DC5_i64);
                // r8 = FNV prime (16777619 = 0x01000193)
                self.emit_mov_reg_imm64(X86Reg::R8, 0x01000193_i64);
                // rdi = 0 (counter)
                self.emit_mov_reg_imm64(X86Reg::Rdi, 0);

                // Loop: while rdi < rcx
                let loop_start = self.text.len();

                // cmp rdi, rcx
                self.emit_cmp_reg_reg(X86Reg::Rdi, X86Reg::Rcx);
                // jge end (forward jump, 2 bytes + rel8)
                self.text.push(0x7D); // jge rel8
                let jge_patch = self.text.len();
                self.text.push(0x00); // placeholder

                // movzx rdx, byte [rsi + rdi]
                // REX.W + movzx rdx, byte [rsi + rdi*1]
                self.text.extend_from_slice(&[0x48, 0x0F, 0xB6, 0x14, 0x3E]);

                // xor eax, edx
                self.text.extend_from_slice(&[0x31, 0xD0]); // xor eax, edx

                // imul eax, r8d (multiply by FNV prime)
                self.text.extend_from_slice(&[0x41, 0x0F, 0xAF, 0xC0]); // imul eax, r8d

                // inc rdi
                self.text.extend_from_slice(&[0x48, 0xFF, 0xC7]); // inc rdi

                // jmp loop_start
                self.text.push(0xEB); // jmp rel8
                let loop_end = self.text.len();
                let offset_back = (loop_start as i64 - loop_end as i64 - 1) as i8;
                self.text.push(offset_back as u8);

                // Patch jge to jump here
                let end_pos = self.text.len();
                self.text[jge_patch] = (end_pos - jge_patch - 1) as u8;

                // Result is in eax (32-bit hash). Store to dest.
                let dest_offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, dest_offset);
            }

            IrInst::FileOpen {
                dest,
                path_offset,
                path_len: _,
            } => {
                // syscall open(path, O_RDONLY, 0)
                // rax = 2 (sys_open), rdi = path addr, rsi = 0 (O_RDONLY), rdx = 0
                // Path is in .data section at data_vaddr + path_offset
                // We use a mov rdi, imm64 that gets patched by linker
                self.emit_mov_reg_imm64(X86Reg::Rdi, *path_offset as i64); // placeholder, patched later
                self.emit_mov_reg_imm64(X86Reg::Rsi, 0); // O_RDONLY
                self.emit_mov_reg_imm64(X86Reg::Rdx, 0);
                self.emit_mov_reg_imm64(X86Reg::Rax, 2); // sys_open
                self.emit_syscall();
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::FileRead { dest, fd, buf, len } => {
                // syscall read(fd, buf, len)
                // rax = 0, rdi = fd, rsi = buf, rdx = len
                self.emit_load_value(X86Reg::Rdi, fd);
                self.emit_load_value(X86Reg::Rsi, buf);
                self.emit_load_value(X86Reg::Rdx, len);
                self.emit_mov_reg_imm64(X86Reg::Rax, 0); // sys_read
                self.emit_syscall();
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::FileClose { fd } => {
                // syscall close(fd)
                // rax = 3, rdi = fd
                self.emit_load_value(X86Reg::Rdi, fd);
                self.emit_mov_reg_imm64(X86Reg::Rax, 3); // sys_close
                self.emit_syscall();
            }

            IrInst::StrCmp { dest, a, b, len } => {
                match a {
                    IrValue::Reg(vreg) => {
                        if self.is_global_vreg(*vreg) {
                            let data_offset = self.global_vregs[vreg];
                            self.emit_load_global_addr(data_offset);
                            self.emit_mov_reg_reg(X86Reg::Rsi, X86Reg::Rax);
                        } else {
                            let off = self.vreg_offset(*vreg);
                            self.emit_lea_rbp_offset(X86Reg::Rsi, off);
                        }
                    }
                    _ => self.emit_load_value(X86Reg::Rsi, a),
                }
                match b {
                    IrValue::Reg(vreg) => {
                        if self.is_global_vreg(*vreg) {
                            let data_offset = self.global_vregs[vreg];
                            self.emit_load_global_addr(data_offset);
                            self.emit_mov_reg_reg(X86Reg::Rdi, X86Reg::Rax);
                        } else {
                            let off = self.vreg_offset(*vreg);
                            self.emit_lea_rbp_offset(X86Reg::Rdi, off);
                        }
                    }
                    _ => self.emit_load_value(X86Reg::Rdi, b),
                }
                self.emit_load_value(X86Reg::Rcx, len);
                self.text.extend_from_slice(&[0x48, 0x85, 0xC9]); // test rcx, rcx
                self.text.extend_from_slice(&[0x74, 0x00]); // jz len_zero (patched below)
                let jz_disp_pos = self.text.len() - 1;
                self.text.push(0xFC); // cld
                self.text.extend_from_slice(&[0xF3, 0xA6]); // repe cmpsb
                self.text.extend_from_slice(&[0x0F, 0x94, 0xC0]); // sete al
                self.text.extend_from_slice(&[0x48, 0x0F, 0xB6, 0xC0]); // movzx rax, al
                self.text.extend_from_slice(&[0xEB, 0x00]); // jmp done (patched below)
                let jmp_done_disp_pos = self.text.len() - 1;
                let len_zero_pos = self.text.len();
                self.emit_mov_reg_imm64(X86Reg::Rax, 1); // len_zero: zero-length strings compare equal
                let done_pos = self.text.len();

                let jz_rel = len_zero_pos as isize - (jz_disp_pos as isize + 1);
                assert!(
                    jz_rel >= i8::MIN as isize && jz_rel <= i8::MAX as isize,
                    "StrCmp jz len_zero out of rel8 range"
                );
                self.text[jz_disp_pos] = jz_rel as i8 as u8;

                let jmp_done_rel = done_pos as isize - (jmp_done_disp_pos as isize + 1);
                assert!(
                    jmp_done_rel >= i8::MIN as isize && jmp_done_rel <= i8::MAX as isize,
                    "StrCmp jmp done out of rel8 range"
                );
                self.text[jmp_done_disp_pos] = jmp_done_rel as i8 as u8;

                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::StrLen { dest, addr } => {
                match addr {
                    IrValue::Reg(vreg) => {
                        if self.is_global_vreg(*vreg) {
                            let data_offset = self.global_vregs[vreg];
                            self.emit_load_global_addr(data_offset);
                            self.emit_mov_reg_reg(X86Reg::Rdi, X86Reg::Rax);
                        } else {
                            let off = self.vreg_offset(*vreg);
                            self.emit_lea_rbp_offset(X86Reg::Rdi, off);
                        }
                    }
                    _ => self.emit_load_value(X86Reg::Rdi, addr),
                }
                self.text.extend_from_slice(&[0x31, 0xC0]); // xor eax, eax
                self.emit_mov_reg_imm64(X86Reg::Rcx, -1_i64); // mov rcx, -1
                self.text.push(0xFC); // cld
                self.text.extend_from_slice(&[0xF2, 0xAE]); // repne scasb
                self.text.extend_from_slice(&[0x48, 0xF7, 0xD1]); // not rcx
                self.text.extend_from_slice(&[0x48, 0xFF, 0xC9]); // dec rcx
                self.emit_mov_reg_reg(X86Reg::Rax, X86Reg::Rcx);
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::StrCopy { dst, src, len } => {
                match dst {
                    IrValue::Reg(vreg) => {
                        if self.is_global_vreg(*vreg) {
                            let data_offset = self.global_vregs[vreg];
                            self.emit_load_global_addr(data_offset);
                            self.emit_mov_reg_reg(X86Reg::Rdi, X86Reg::Rax);
                        } else {
                            let off = self.vreg_offset(*vreg);
                            self.emit_lea_rbp_offset(X86Reg::Rdi, off);
                        }
                    }
                    _ => self.emit_load_value(X86Reg::Rdi, dst),
                }
                match src {
                    IrValue::Reg(vreg) => {
                        if self.is_global_vreg(*vreg) {
                            let data_offset = self.global_vregs[vreg];
                            self.emit_load_global_addr(data_offset);
                            self.emit_mov_reg_reg(X86Reg::Rsi, X86Reg::Rax);
                        } else {
                            let off = self.vreg_offset(*vreg);
                            self.emit_lea_rbp_offset(X86Reg::Rsi, off);
                        }
                    }
                    _ => self.emit_load_value(X86Reg::Rsi, src),
                }
                self.emit_load_value(X86Reg::Rcx, len);
                self.text.push(0xFC); // cld
                self.text.extend_from_slice(&[0xF3, 0xA4]); // rep movsb
            }

            IrInst::Nop => {
                self.emit_nop();
            }

            // ── Kernel / bare-metal instructions ──

            IrInst::Outb { port, value } => {
                // out dx, al — 0xEE (byte from al to port in dx)
                self.emit_load_value(X86Reg::Rdx, port);
                self.emit_load_value(X86Reg::Rax, value);
                self.text.push(0xEE); // out dx, al
            }

            IrInst::Inb { dest, port } => {
                // in al, dx — 0xEC (byte from port in dx to al)
                self.emit_load_value(X86Reg::Rdx, port);
                self.text.push(0xEC); // in al, dx
                self.text.extend_from_slice(&[0x48, 0x0F, 0xB6, 0xC0]); // movzx rax, al
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Cli => {
                self.text.push(0xFA); // cli
            }

            IrInst::Sti => {
                self.text.push(0xFB); // sti
            }

            IrInst::Lgdt { addr } => {
                // lgdt [rax] — load GDT descriptor from address in rax
                self.emit_load_value(X86Reg::Rax, addr);
                self.text.extend_from_slice(&[0x0F, 0x01, 0x10]); // lgdt [rax]
            }

            IrInst::Lidt { addr } => {
                // lidt [rax] — load IDT descriptor from address in rax
                self.emit_load_value(X86Reg::Rax, addr);
                self.text.extend_from_slice(&[0x0F, 0x01, 0x18]); // lidt [rax]
            }

            IrInst::Wrmsr { msr, value } => {
                // wrmsr: ecx=MSR number, edx:eax=value
                self.emit_load_value(X86Reg::Rax, value);
                // Split 64-bit value: edx = high 32, eax = low 32
                self.text.extend_from_slice(&[0x48, 0x89, 0xC2]); // mov rdx, rax
                self.text.extend_from_slice(&[0x48, 0xC1, 0xEA, 0x20]); // shr rdx, 32
                self.emit_load_value(X86Reg::Rcx, msr);
                self.text.extend_from_slice(&[0x0F, 0x30]); // wrmsr
            }

            IrInst::Rdmsr { dest, msr } => {
                // rdmsr: ecx=MSR number, result in edx:eax
                self.emit_load_value(X86Reg::Rcx, msr);
                self.text.extend_from_slice(&[0x0F, 0x32]); // rdmsr
                // Combine edx:eax into rax (rax = (rdx << 32) | eax)
                self.text.extend_from_slice(&[0x48, 0xC1, 0xE2, 0x20]); // shl rdx, 32
                self.text.extend_from_slice(&[0x48, 0x09, 0xD0]); // or rax, rdx
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Invlpg { addr } => {
                // invlpg [rax]
                self.emit_load_value(X86Reg::Rax, addr);
                self.text.extend_from_slice(&[0x0F, 0x01, 0x38]); // invlpg [rax]
            }

            IrInst::WriteCr { cr, value } => {
                // mov crN, rax — CR number must be immediate
                self.emit_load_value(X86Reg::Rax, value);
                if let IrValue::Imm(n) = cr {
                    match n {
                        0 => self.text.extend_from_slice(&[0x0F, 0x22, 0xC0]), // mov cr0, rax
                        3 => self.text.extend_from_slice(&[0x0F, 0x22, 0xD8]), // mov cr3, rax
                        4 => self.text.extend_from_slice(&[0x0F, 0x22, 0xE0]), // mov cr4, rax
                        _ => self.text.extend_from_slice(&[0x0F, 0x0B]),       // ud2
                    }
                }
            }

            IrInst::ReadCr { dest, cr } => {
                // mov rax, crN
                if let IrValue::Imm(n) = cr {
                    match n {
                        0 => self.text.extend_from_slice(&[0x0F, 0x20, 0xC0]), // mov rax, cr0
                        2 => self.text.extend_from_slice(&[0x0F, 0x20, 0xD0]), // mov rax, cr2
                        3 => self.text.extend_from_slice(&[0x0F, 0x20, 0xD8]), // mov rax, cr3
                        4 => self.text.extend_from_slice(&[0x0F, 0x20, 0xE0]), // mov rax, cr4
                        _ => self.text.extend_from_slice(&[0x0F, 0x0B]),       // ud2
                    }
                }
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }

            IrInst::Iretq => {
                self.text.extend_from_slice(&[0x48, 0xCF]); // iretq (REX.W + iret)
            }

            IrInst::Ltr { selector } => {
                // ltr ax — load task register
                self.emit_load_value(X86Reg::Rax, selector);
                self.text.extend_from_slice(&[0x0F, 0x00, 0xD8]); // ltr ax
            }

            IrInst::StoreAbs { addr, value, ty } => {
                // Write value to absolute address
                self.emit_load_value(X86Reg::Rax, addr);
                self.emit_load_value(X86Reg::Rcx, value);
                match ty {
                    JStarType::Byte => {
                        self.text.extend_from_slice(&[0x88, 0x08]); // mov [rax], cl
                    }
                    _ => {
                        self.text.extend_from_slice(&[0x48, 0x89, 0x08]); // mov [rax], rcx
                    }
                }
            }

            IrInst::LoadAbs { dest, addr, ty } => {
                // Read value from absolute address
                self.emit_load_value(X86Reg::Rax, addr);
                match ty {
                    JStarType::Byte => {
                        // movzx rax, byte [rax]
                        self.text.extend_from_slice(&[0x48, 0x0F, 0xB6, 0x00]);
                    }
                    _ => {
                        // mov rax, [rax]
                        self.text.extend_from_slice(&[0x48, 0x8B, 0x00]);
                    }
                }
                let offset = self.vreg_offset(*dest);
                self.emit_store_reg_to_rbp_offset(X86Reg::Rax, offset);
            }
        }
        Ok(())
    }

    fn emit_terminator(&mut self, term: &Terminator) -> MorphResult<()> {
        match term {
            Terminator::Return(value) => {
                // Load return value into rax
                if let Some(val) = value {
                    self.emit_load_value(X86Reg::Rax, val);
                }
                if self.is_entry_point {
                    // _start: exit via syscall
                    self.emit_mov_reg_reg(X86Reg::Rdi, X86Reg::Rax);
                    self.emit_mov_reg_imm64(X86Reg::Rax, 60);
                    self.emit_syscall();
                } else {
                    // Regular function: epilogue + ret
                    self.emit_mov_reg_reg(X86Reg::Rsp, X86Reg::Rbp);
                    self.emit_pop_reg(X86Reg::Rbp);
                    self.text.push(0xC3); // ret
                }
            }

            Terminator::Halt(code) => {
                if self.is_entry_point {
                    self.emit_load_value(X86Reg::Rdi, code);
                    self.emit_mov_reg_imm64(X86Reg::Rax, 60);
                    self.emit_syscall();
                } else {
                    // Non-_start: treat halt as return 0
                    self.emit_mov_reg_imm64(X86Reg::Rax, 0);
                    self.emit_mov_reg_reg(X86Reg::Rsp, X86Reg::Rbp);
                    self.emit_pop_reg(X86Reg::Rbp);
                    self.text.push(0xC3); // ret
                }
            }

            Terminator::Jump(label) => {
                // jmp rel32 (0xE9 + 4-byte signed displacement)
                self.text.push(0xE9);
                let patch_offset = self.text.len();
                self.text.extend_from_slice(&0i32.to_le_bytes()); // placeholder
                self.fixups.push((patch_offset, label.clone()));
            }

            Terminator::Branch {
                cond,
                true_label,
                false_label,
            } => {
                // Load condition vreg into rax
                let offset = self.vreg_offset(*cond);
                self.emit_load_from_rbp_offset(X86Reg::Rax, offset);
                // test rax, rax (sets ZF if rax == 0)
                self.text.extend_from_slice(&[0x48, 0x85, 0xC0]);
                // jnz true_label (0x0F 0x85 rel32) — jump if nonzero
                self.text.extend_from_slice(&[0x0F, 0x85]);
                let patch_true = self.text.len();
                self.text.extend_from_slice(&0i32.to_le_bytes()); // placeholder
                self.fixups.push((patch_true, true_label.clone()));
                // jmp false_label (0xE9 rel32) — fall through to false
                self.text.push(0xE9);
                let patch_false = self.text.len();
                self.text.extend_from_slice(&0i32.to_le_bytes()); // placeholder
                self.fixups.push((patch_false, false_label.clone()));
            }

            Terminator::Unreachable => {
                // UD2 — undefined instruction trap
                self.text.extend_from_slice(&[0x0F, 0x0B]);
            }
        }
        Ok(())
    }

    /// Emit x86-64 code to print an integer value to stdout as decimal ASCII + newline.
    ///
    /// Algorithm: load value into rax, convert to decimal digits in a stack buffer
    /// (right-to-left via repeated unsigned div by 10), append newline, sys_write.
    ///
    /// Uses 32 bytes of stack space temporarily. Clobbers rax, rcx, rdx, rsi, rdi.
    /// All caller-saved — no preservation needed.
    ///
    /// The routine is 78 bytes of fixed x86-64 with precomputed internal jump offsets:
    ///   .zero_check → .convert (skip zero case)
    ///   .zero_case  → .write  (skip convert loop)
    ///   .convert    → .convert (back-edge: more digits?)
    fn emit_print_integer(&mut self, value: &IrValue) {
        // Load the value to print into rax
        self.emit_load_value(X86Reg::Rax, value);

        // The print routine as hand-encoded x86-64.
        // See codegen.rs doc comments for the full assembly listing.
        //
        // Layout:
        //   [rsp+0..29]  digit buffer (filled right-to-left)
        //   [rsp+30]     last digit position
        //   [rsp+31]     newline (0x0A)
        //
        // Registers during routine:
        //   rax = quotient (shrinks toward 0)
        //   rcx = 10 (divisor)
        //   rdx = remainder (becomes digit)
        //   rsi = write pointer (moves left)
        //   rdi = fd (1 = stdout, set before syscall)
        let print_code: [u8; 78] = [
            // sub rsp, 32
            0x48, 0x83, 0xEC, 0x20, // mov byte [rsp+31], 0x0A (newline)
            0xC6, 0x44, 0x24, 0x1F, 0x0A,
            // lea rsi, [rsp+30] (write pointer = rightmost digit slot)
            0x48, 0x8D, 0x74, 0x24, 0x1E, // mov ecx, 10 (divisor, zero-extends to rcx)
            0xB9, 0x0A, 0x00, 0x00, 0x00, // test rax, rax (zero check)
            0x48, 0x85, 0xC0, // jnz .convert (+8 bytes, skip zero case)
            0x75, 0x08, // --- zero case ---
            // mov byte [rsi], 0x30 ('0')
            0xC6, 0x06, 0x30, // dec rsi
            0x48, 0xFF, 0xCE, // jmp .write (+0x13 bytes)
            0xEB, 0x13,
            // --- .convert (itoa loop) ---
            // xor rdx, rdx (clear for unsigned div)
            0x48, 0x31, 0xD2, // div rcx (rax = rax/10, rdx = rax%10)
            0x48, 0xF7, 0xF1, // add dl, 0x30 (remainder -> ASCII digit)
            0x80, 0xC2, 0x30, // mov [rsi], dl (store digit)
            0x88, 0x16, // dec rsi (move write pointer left)
            0x48, 0xFF, 0xCE, // test rax, rax (more digits?)
            0x48, 0x85, 0xC0, // jnz .convert (-0x13 bytes, back-edge)
            0x75, 0xED, // --- .write ---
            // inc rsi (point to first digit)
            0x48, 0xFF, 0xC6, // lea rdx, [rsp+32] (end of buffer, past newline)
            0x48, 0x8D, 0x54, 0x24, 0x20, // sub rdx, rsi (length = end - start)
            0x48, 0x29, 0xF2, // mov edi, 1 (fd = stdout)
            0xBF, 0x01, 0x00, 0x00, 0x00, // mov eax, 1 (sys_write)
            0xB8, 0x01, 0x00, 0x00, 0x00, // syscall
            0x0F, 0x05, // add rsp, 32 (deallocate buffer)
            0x48, 0x83, 0xC4, 0x20,
        ];
        self.text.extend_from_slice(&print_code);
    }

    /// Load an IrValue into a register.
    fn emit_load_value(&mut self, dest: X86Reg, value: &IrValue) {
        match value {
            IrValue::Imm(imm) => {
                self.emit_mov_reg_imm64(dest, *imm);
            }
            IrValue::Reg(vreg) => {
                if let Some(&data_offset) = self.global_vregs.get(vreg) {
                    // Global variable: load from .data section via absolute address.
                    // Use dest register itself as temp (movabs dest, addr; mov dest, [dest]).
                    // This avoids clobbering rax when loading into rcx/rdx/etc.
                    self.emit_load_global_into(dest, data_offset);
                } else {
                    let offset = self.vreg_offset(*vreg);
                    self.emit_load_from_rbp_offset(dest, offset);
                }
            }
            IrValue::Named(_name) => {
                // Named variables: for now, load 0 (resolved later with symbol table)
                self.emit_mov_reg_imm64(dest, 0);
            }
        }
    }

    // ─── Global Variable Helpers ────────────────────────────────────────────

    /// Check if a vreg is a global variable (lives in .data, not on stack).
    fn is_global_vreg(&self, vreg: VReg) -> bool {
        self.global_vregs.contains_key(&vreg)
    }

    /// Whether a vreg names inline storage directly rather than a slot containing a value.
    fn is_direct_storage_vreg(&self, vreg: VReg) -> bool {
        self.direct_storage_vregs.contains(&vreg)
    }

    /// Whether an instruction reserves inline storage that should be addressed directly.
    fn instruction_allocates_direct_storage(&self, inst: &IrInst) -> bool {
        match inst {
            IrInst::Alloca { size, ty, .. } => *size > ty.size_bytes(),
            IrInst::ArrayAlloc { .. } => true,
            _ => false,
        }
    }

    /// Load a base pointer for indexed or hashing operations.
    ///
    /// Direct-storage vregs become their address.
    /// Pointer/scalar vregs load the value stored in the slot/global.
    fn emit_index_base(&mut self, dest: X86Reg, base: VReg) {
        if self.is_direct_storage_vreg(base) {
            if self.is_global_vreg(base) {
                let data_offset = self.global_vregs[&base];
                self.emit_load_global_addr_into(dest, data_offset);
            } else {
                let base_offset = self.vreg_offset(base);
                self.emit_lea_rbp_offset(dest, base_offset);
            }
        } else if let Some(&data_offset) = self.global_vregs.get(&base) {
            self.emit_load_global_into(dest, data_offset);
        } else {
            let base_offset = self.vreg_offset(base);
            self.emit_load_from_rbp_offset(dest, base_offset);
        }
    }

    fn emit_load_indexable_base(&mut self, dest: X86Reg, base: &IrValue) {
        match base {
            IrValue::Reg(vreg) => self.emit_index_base(dest, *vreg),
            _ => self.emit_load_value(dest, base),
        }
    }

    /// Emit: movabs rax, <data_offset>  (absolute address placeholder)
    ///       mov rax, [rax]              (dereference to load the value)
    /// The linker adds data_vaddr to the placeholder.
    fn emit_load_global(&mut self, data_offset: usize) {
        // movabs rax, imm64 (address placeholder)
        self.text.push(0x48); // REX.W
        self.text.push(0xB8); // movabs rax
        let fixup_pos = self.text.len();
        self.text
            .extend_from_slice(&(data_offset as u64).to_le_bytes());
        self.data_fixups.push(fixup_pos);
        // mov rax, [rax]
        self.text.push(0x48); // REX.W
        self.text.push(0x8B); // MOV r64, r/m64
        self.text.push(0x00); // ModRM: mod=00, reg=rax(000), rm=rax(000) = [rax]
    }

    /// Emit: movabs dest, <data_offset>  (absolute address into dest register)
    ///       mov dest, [dest]             (dereference to load the value)
    /// Uses the dest register as temp to avoid clobbering other registers.
    fn emit_load_global_into(&mut self, dest: X86Reg, data_offset: usize) {
        // movabs dest, imm64
        let mut rex: u8 = 0x48; // REX.W
        if dest.needs_rex_ext() {
            rex |= 0x01; // REX.B
        }
        self.text.push(rex);
        self.text.push(0xB8 + dest.encoding()); // movabs dest
        let fixup_pos = self.text.len();
        self.text
            .extend_from_slice(&(data_offset as u64).to_le_bytes());
        self.data_fixups.push(fixup_pos);
        // mov dest, [dest] — dereference
        let mut rex2: u8 = 0x48; // REX.W
        if dest.needs_rex_ext() {
            rex2 |= 0x04; // REX.R (reg field)
            rex2 |= 0x01; // REX.B (rm field)
        }
        self.text.push(rex2);
        self.text.push(0x8B); // MOV r64, r/m64
        // ModRM: mod=00, reg=dest, rm=dest = [dest]
        self.text.push((dest.encoding() << 3) | dest.encoding());
    }

    /// Emit: movabs rax, <data_offset>  (absolute address placeholder)
    ///       mov [rax], rcx              (store rcx to global)
    /// Value to store must be in rcx before calling this.
    fn emit_store_global(&mut self, data_offset: usize) {
        // movabs rax, imm64 (address placeholder)
        self.text.push(0x48); // REX.W
        self.text.push(0xB8); // movabs rax
        let fixup_pos = self.text.len();
        self.text
            .extend_from_slice(&(data_offset as u64).to_le_bytes());
        self.data_fixups.push(fixup_pos);
        // mov [rax], rcx
        self.text.push(0x48); // REX.W
        self.text.push(0x89); // MOV r/m64, r64
        self.text.push(0x08); // ModRM: mod=00, reg=rcx(001), rm=rax(000) = [rax]
    }

    /// Emit: movabs rax, <data_offset>  (absolute address placeholder)
    ///       mov byte [rax], cl          (store byte to global)
    fn emit_store_byte_global(&mut self, data_offset: usize) {
        // movabs rax, imm64 (address placeholder)
        self.text.push(0x48); // REX.W
        self.text.push(0xB8); // movabs rax
        let fixup_pos = self.text.len();
        self.text
            .extend_from_slice(&(data_offset as u64).to_le_bytes());
        self.data_fixups.push(fixup_pos);
        // mov byte [rax], cl
        self.text.push(0x88); // MOV r/m8, r8
        self.text.push(0x08); // ModRM: mod=00, reg=cl(001), rm=rax(000) = [rax]
    }

    /// Emit: movabs rax, <data_offset>  (absolute address placeholder)
    ///       movzx rax, byte [rax]       (load byte from global, zero-extend)
    fn emit_load_byte_global(&mut self, data_offset: usize) {
        // movabs rax, imm64 (address placeholder)
        self.text.push(0x48); // REX.W
        self.text.push(0xB8); // movabs rax
        let fixup_pos = self.text.len();
        self.text
            .extend_from_slice(&(data_offset as u64).to_le_bytes());
        self.data_fixups.push(fixup_pos);
        // movzx rax, byte [rax]
        self.text.push(0x48); // REX.W
        self.text.push(0x0F);
        self.text.push(0xB6); // MOVZX r64, r/m8
        self.text.push(0x00); // ModRM: mod=00, reg=rax(000), rm=rax(000) = [rax]
    }

    /// Emit: movabs rax, <data_offset>  (absolute address, no dereference)
    /// Used for addressof and as base address for indexed access.
    fn emit_load_global_addr(&mut self, data_offset: usize) {
        self.emit_load_global_addr_into(X86Reg::Rax, data_offset);
    }

    /// Emit: movabs dest, <data_offset>  (absolute address, no dereference)
    fn emit_load_global_addr_into(&mut self, dest: X86Reg, data_offset: usize) {
        let mut rex: u8 = 0x48; // REX.W
        if dest.needs_rex_ext() {
            rex |= 0x01; // REX.B
        }
        self.text.push(rex);
        self.text.push(0xB8 + dest.encoding()); // movabs dest
        let fixup_pos = self.text.len();
        self.text
            .extend_from_slice(&(data_offset as u64).to_le_bytes());
        self.data_fixups.push(fixup_pos);
    }

    // ─── x86-64 Instruction Encoding ────────────────────────────────────────

    /// REX prefix byte for 64-bit operand size.
    fn rex_w(reg: X86Reg, rm: X86Reg) -> u8 {
        let mut rex: u8 = 0x48; // REX.W
        if reg.needs_rex_ext() {
            rex |= 0x04; // REX.R
        }
        if rm.needs_rex_ext() {
            rex |= 0x01; // REX.B
        }
        rex
    }

    /// ModR/M byte for register-register addressing.
    fn modrm_reg(reg: X86Reg, rm: X86Reg) -> u8 {
        0xC0 | (reg.encoding() << 3) | rm.encoding()
    }

    /// ModR/M byte for [rbp+disp32] addressing.
    fn modrm_rbp_disp32(reg: X86Reg) -> u8 {
        0x80 | (reg.encoding() << 3) | X86Reg::Rbp.encoding()
    }

    /// push reg
    fn emit_push_reg(&mut self, reg: X86Reg) {
        if reg.needs_rex_ext() {
            self.text.push(0x41); // REX.B
        }
        self.text.push(0x50 + reg.encoding());
    }

    /// pop reg
    fn emit_pop_reg(&mut self, reg: X86Reg) {
        if reg.needs_rex_ext() {
            self.text.push(0x41);
        }
        self.text.push(0x58 + reg.encoding());
    }

    /// mov reg, reg (64-bit)
    fn emit_mov_reg_reg(&mut self, dest: X86Reg, src: X86Reg) {
        self.text.push(Self::rex_w(src, dest));
        self.text.push(0x89); // MOV r/m64, r64
        self.text.push(Self::modrm_reg(src, dest));
    }

    /// mov reg, imm64 (movabs)
    fn emit_mov_reg_imm64(&mut self, dest: X86Reg, imm: i64) {
        // Optimization: use 32-bit mov if value fits
        if imm >= 0 && imm <= u32::MAX as i64 {
            if dest.needs_rex_ext() {
                self.text.push(0x41); // REX.B
            }
            self.text.push(0xB8 + dest.encoding()); // MOV r32, imm32
            self.text.extend_from_slice(&(imm as u32).to_le_bytes());
        } else {
            let mut rex: u8 = 0x48; // REX.W
            if dest.needs_rex_ext() {
                rex |= 0x01; // REX.B
            }
            self.text.push(rex);
            self.text.push(0xB8 + dest.encoding()); // MOV r64, imm64
            self.text.extend_from_slice(&imm.to_le_bytes());
        }
    }

    /// add reg, reg (64-bit)
    fn emit_add_reg_reg(&mut self, dest: X86Reg, src: X86Reg) {
        self.text.push(Self::rex_w(src, dest));
        self.text.push(0x01); // ADD r/m64, r64
        self.text.push(Self::modrm_reg(src, dest));
    }

    /// sub reg, reg (64-bit)
    fn emit_sub_reg_reg(&mut self, dest: X86Reg, src: X86Reg) {
        self.text.push(Self::rex_w(src, dest));
        self.text.push(0x29); // SUB r/m64, r64
        self.text.push(Self::modrm_reg(src, dest));
    }

    /// sub reg, imm32 (64-bit)
    fn emit_sub_reg_imm(&mut self, reg: X86Reg, imm: i32) {
        let mut rex: u8 = 0x48;
        if reg.needs_rex_ext() {
            rex |= 0x01;
        }
        self.text.push(rex);
        self.text.push(0x81); // SUB r/m64, imm32
        self.text.push(0xC0 | (5 << 3) | reg.encoding()); // /5 = SUB
        self.text.extend_from_slice(&imm.to_le_bytes());
    }

    /// imul reg, reg (64-bit signed multiply)
    fn emit_imul_reg_reg(&mut self, dest: X86Reg, src: X86Reg) {
        self.text.push(Self::rex_w(dest, src));
        self.text.push(0x0F);
        self.text.push(0xAF); // IMUL r64, r/m64
        self.text.push(Self::modrm_reg(dest, src));
    }

    /// cqo — sign-extend rax into rdx:rax
    fn emit_cqo(&mut self) {
        self.text.push(0x48); // REX.W
        self.text.push(0x99); // CQO
    }

    /// idiv r/m64 — signed divide rdx:rax by reg
    fn emit_idiv(&mut self, divisor: X86Reg) {
        let mut rex: u8 = 0x48;
        if divisor.needs_rex_ext() {
            rex |= 0x01;
        }
        self.text.push(rex);
        self.text.push(0xF7); // IDIV r/m64
        self.text.push(0xC0 | (7 << 3) | divisor.encoding()); // /7 = IDIV
    }

    /// and reg, reg
    fn emit_and_reg_reg(&mut self, dest: X86Reg, src: X86Reg) {
        self.text.push(Self::rex_w(src, dest));
        self.text.push(0x21);
        self.text.push(Self::modrm_reg(src, dest));
    }

    /// or reg, reg
    fn emit_or_reg_reg(&mut self, dest: X86Reg, src: X86Reg) {
        self.text.push(Self::rex_w(src, dest));
        self.text.push(0x09);
        self.text.push(Self::modrm_reg(src, dest));
    }

    /// xor reg, reg
    fn emit_xor_reg_reg(&mut self, dest: X86Reg, src: X86Reg) {
        self.text.push(Self::rex_w(src, dest));
        self.text.push(0x31);
        self.text.push(Self::modrm_reg(src, dest));
    }

    /// shl reg, cl
    fn emit_shl_reg_cl(&mut self, reg: X86Reg) {
        let mut rex: u8 = 0x48;
        if reg.needs_rex_ext() {
            rex |= 0x01;
        }
        self.text.push(rex);
        self.text.push(0xD3); // SHL r/m64, CL
        self.text.push(0xC0 | (4 << 3) | reg.encoding()); // /4 = SHL
    }

    /// shr reg, cl
    fn emit_shr_reg_cl(&mut self, reg: X86Reg) {
        let mut rex: u8 = 0x48;
        if reg.needs_rex_ext() {
            rex |= 0x01;
        }
        self.text.push(rex);
        self.text.push(0xD3);
        self.text.push(0xC0 | (5 << 3) | reg.encoding()); // /5 = SHR
    }

    /// neg reg (two's complement negate)
    fn emit_neg(&mut self, reg: X86Reg) {
        let mut rex: u8 = 0x48;
        if reg.needs_rex_ext() {
            rex |= 0x01;
        }
        self.text.push(rex);
        self.text.push(0xF7);
        self.text.push(0xC0 | (3 << 3) | reg.encoding()); // /3 = NEG
    }

    /// not reg (bitwise complement)
    fn emit_not(&mut self, reg: X86Reg) {
        let mut rex: u8 = 0x48;
        if reg.needs_rex_ext() {
            rex |= 0x01;
        }
        self.text.push(rex);
        self.text.push(0xF7);
        self.text.push(0xC0 | (2 << 3) | reg.encoding()); // /2 = NOT
    }

    /// cmp reg, reg
    fn emit_cmp_reg_reg(&mut self, lhs: X86Reg, rhs: X86Reg) {
        self.text.push(Self::rex_w(rhs, lhs));
        self.text.push(0x39);
        self.text.push(Self::modrm_reg(rhs, lhs));
    }

    /// setcc al; movzx rax, al — set rax to 0 or 1 based on CmpKind.
    fn emit_setcc(&mut self, kind: CmpKind) {
        // All setcc instructions are 0x0F followed by the condition opcode, then ModR/M for al (0xC0)
        let opcode = match kind {
            CmpKind::Eq => 0x94, // sete
            CmpKind::Ne => 0x95, // setne
            CmpKind::Lt => 0x9C, // setl
            CmpKind::Le => 0x9E, // setle
            CmpKind::Gt => 0x9F, // setg
            CmpKind::Ge => 0x9D, // setge
        };
        self.text.extend_from_slice(&[0x0F, opcode, 0xC0]);
        // movzx rax, al (zero-extend al into full rax)
        self.text.extend_from_slice(&[0x48, 0x0F, 0xB6, 0xC0]);
    }

    /// Resolve all jump fixups after all blocks have been emitted.
    ///
    /// Each fixup is (patch_offset, target_label). The displacement is
    /// target_address - (patch_offset + 4), because the CPU reads IP
    /// past the displacement field before adding it.
    ///
    /// Returns an error if any target label was never emitted as a block.
    /// Previously this defaulted to offset 0, silently corrupting the
    /// function prologue.
    fn apply_fixups(&mut self) -> MorphResult<()> {
        for (patch_offset, target_label) in &self.fixups {
            let target = *self.label_offsets.get(target_label).ok_or_else(|| {
                MorphlexError::CodegenError(format!(
                    "missing label '{}' at fixup offset {} (known labels: {:?})",
                    target_label,
                    patch_offset,
                    self.label_offsets.keys().collect::<Vec<_>>()
                ))
            })?;
            let disp = (target as i32) - (*patch_offset as i32 + 4);
            let bytes = disp.to_le_bytes();
            self.text[*patch_offset] = bytes[0];
            self.text[*patch_offset + 1] = bytes[1];
            self.text[*patch_offset + 2] = bytes[2];
            self.text[*patch_offset + 3] = bytes[3];
        }
        self.fixups.clear();
        Ok(())
    }

    /// Resolve all call fixups after all functions have been emitted.
    ///
    /// Returns an error if any called function was never emitted.
    fn apply_call_fixups(&mut self) -> MorphResult<()> {
        for (patch_offset, func_name) in &self.call_fixups {
            let target = *self.function_offsets.get(func_name).ok_or_else(|| {
                MorphlexError::CodegenError(format!(
                    "missing function '{}' at call fixup offset {} \
                     (known functions: {:?})",
                    func_name,
                    patch_offset,
                    self.function_offsets.keys().collect::<Vec<_>>()
                ))
            })?;
            let disp = (target as i32) - (*patch_offset as i32 + 4);
            let bytes = disp.to_le_bytes();
            self.text[*patch_offset] = bytes[0];
            self.text[*patch_offset + 1] = bytes[1];
            self.text[*patch_offset + 2] = bytes[2];
            self.text[*patch_offset + 3] = bytes[3];
        }
        self.call_fixups.clear();
        Ok(())
    }

    /// Emit x86-64 code to print a string from the .data section.
    ///
    /// Uses sys_write(1, data_vaddr + offset, len).
    /// The absolute address is computed at link time — we emit a movabs
    /// with a placeholder that the linker patches via data_fixups.
    fn emit_print_string(&mut self, data_offset: usize, len: usize) {
        // mov eax, 1 (sys_write)
        self.text.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]);
        // mov edi, 1 (fd = stdout)
        self.text.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]);
        // mov rsi, <data_offset> — placeholder, linker adds data_vaddr
        self.text.push(0x48); // REX.W
        self.text.push(0xBE); // MOV rsi, imm64
        let fixup_pos = self.text.len();
        self.text
            .extend_from_slice(&(data_offset as u64).to_le_bytes());
        self.data_fixups.push(fixup_pos);
        // mov edx, len
        self.text.push(0xBA); // MOV edx, imm32
        self.text.extend_from_slice(&(len as u32).to_le_bytes());
        // syscall
        self.text.extend_from_slice(&[0x0F, 0x05]);
    }

    /// mov reg, [rax] (load indirect)
    fn emit_load_indirect(&mut self, dest: X86Reg, addr: X86Reg) {
        self.text.push(Self::rex_w(dest, addr));
        self.text.push(0x8B); // MOV r64, r/m64
        self.text.push((dest.encoding() << 3) | addr.encoding()); // ModR/M [addr]
    }

    /// mov [addr], src (store indirect)
    fn emit_store_indirect(&mut self, addr: X86Reg, src: X86Reg) {
        self.text.push(Self::rex_w(src, addr));
        self.text.push(0x89);
        self.text.push((src.encoding() << 3) | addr.encoding());
    }

    /// mov reg, [rbp+offset]
    fn emit_load_from_rbp_offset(&mut self, dest: X86Reg, offset: i32) {
        self.text.push(Self::rex_w(dest, X86Reg::Rbp));
        self.text.push(0x8B); // MOV r64, r/m64
        self.text.push(Self::modrm_rbp_disp32(dest));
        self.text.extend_from_slice(&offset.to_le_bytes());
    }

    /// mov [rbp+offset], reg
    fn emit_store_reg_to_rbp_offset(&mut self, src: X86Reg, offset: i32) {
        self.text.push(Self::rex_w(src, X86Reg::Rbp));
        self.text.push(0x89); // MOV r/m64, r64
        self.text.push(Self::modrm_rbp_disp32(src));
        self.text.extend_from_slice(&offset.to_le_bytes());
    }

    /// lea reg, [rbp+offset]
    fn emit_lea_rbp_offset(&mut self, dest: X86Reg, offset: i32) {
        self.text.push(Self::rex_w(dest, X86Reg::Rbp));
        self.text.push(0x8D); // LEA r64, m
        self.text.push(Self::modrm_rbp_disp32(dest));
        self.text.extend_from_slice(&offset.to_le_bytes());
    }

    /// syscall
    fn emit_syscall(&mut self) {
        self.text.extend_from_slice(&[0x0F, 0x05]);
    }

    /// mov byte [rbp+offset], src_low8 — store a single byte to stack slot
    fn emit_store_byte_to_rbp_offset(&mut self, src: X86Reg, offset: i32) {
        // For r8-r15 we need REX prefix even for byte access
        let mut rex: u8 = 0x40; // REX (no W — byte operation)
        if src.needs_rex_ext() {
            rex |= 0x04; // REX.R
        }
        // REX prefix is needed for SPL/BPL/SIL/DIL access or extended regs
        // Always emit it to be safe with any register
        self.text.push(rex);
        self.text.push(0x88); // MOV r/m8, r8
        self.text.push(Self::modrm_rbp_disp32(src));
        self.text.extend_from_slice(&offset.to_le_bytes());
    }

    /// movzx reg64, byte [rbp+offset] — load a single byte, zero-extend to 64 bits
    fn emit_load_byte_from_rbp_offset(&mut self, dest: X86Reg, offset: i32) {
        self.text.push(Self::rex_w(dest, X86Reg::Rbp));
        self.text.push(0x0F);
        self.text.push(0xB6); // MOVZX r64, r/m8
        self.text.push(Self::modrm_rbp_disp32(dest));
        self.text.extend_from_slice(&offset.to_le_bytes());
    }

    /// Check if a JStarType should use byte-width operations.
    fn is_byte_type(ty: &JStarType) -> bool {
        matches!(ty, JStarType::Byte | JStarType::Boolean | JStarType::Char)
    }

    /// nop
    fn emit_nop(&mut self) {
        self.text.push(0x90);
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

/// Write crash report for debugging self-host divergence
impl CodeGen {
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_rbp_disp32_insn(text: &[u8], opcode: u8, offset: i32) -> bool {
        let mut pattern = vec![0x48, opcode, 0x85];
        pattern.extend_from_slice(&offset.to_le_bytes());
        text.windows(pattern.len()).any(|w| w == pattern.as_slice())
    }
    use super::super::grammar::JStarType;

    #[test]
    fn test_generate_empty_program() {
        let ir = IrProgram {
            string_data: vec![],
            global_data: vec![],
            global_vars: std::collections::HashMap::new(),
            global_vregs: std::collections::HashMap::new(),
            functions: vec![IrFunction {
                name: "_start".to_string(),
                return_type: JStarType::Int,
                param_vregs: vec![],
                blocks: vec![BasicBlock {
                    label: "entry".to_string(),
                    instructions: vec![],
                    terminator: Terminator::Halt(IrValue::Imm(0)),
                }],
                next_vreg: 0,
                param_count: 0,
            }],
        };
        let mc = generate(&ir).unwrap();
        assert!(!mc.text.is_empty(), "Should generate prologue + halt code");
    }

    #[test]
    fn test_generate_return_42() {
        let ir = IrProgram {
            string_data: vec![],
            global_data: vec![],
            global_vars: std::collections::HashMap::new(),
            global_vregs: std::collections::HashMap::new(),
            functions: vec![IrFunction {
                name: "_start".to_string(),
                return_type: JStarType::Int,
                param_vregs: vec![],
                blocks: vec![BasicBlock {
                    label: "entry".to_string(),
                    instructions: vec![],
                    terminator: Terminator::Return(Some(IrValue::Imm(42))),
                }],
                next_vreg: 0,
                param_count: 0,
            }],
        };
        let mc = generate(&ir).unwrap();
        // The code should contain:
        // mov rdi, 42 (exit code)
        // mov rax, 60 (sys_exit)
        // syscall
        assert!(mc.text.len() > 10, "Should generate meaningful code");
        // Check syscall instruction is present (0x0F 0x05)
        let has_syscall = mc.text.windows(2).any(|w| w == [0x0F, 0x05]);
        assert!(has_syscall, "Should contain syscall instruction");
    }

    #[test]
    fn test_codegen_determinism() {
        let ir = IrProgram {
            string_data: vec![],
            global_data: vec![],
            global_vars: std::collections::HashMap::new(),
            global_vregs: std::collections::HashMap::new(),
            functions: vec![IrFunction {
                name: "_start".to_string(),
                return_type: JStarType::Int,
                param_vregs: vec![],
                blocks: vec![BasicBlock {
                    label: "entry".to_string(),
                    instructions: vec![IrInst::BinOp {
                        dest: 0,
                        op: IrBinOp::Add,
                        lhs: IrValue::Imm(1),
                        rhs: IrValue::Imm(2),
                        ty: JStarType::Int,
                    }],
                    terminator: Terminator::Halt(IrValue::Imm(0)),
                }],
                next_vreg: 1,
                param_count: 0,
            }],
        };
        let a = generate(&ir).unwrap();
        let b = generate(&ir).unwrap();
        assert_eq!(a.text, b.text, "Codegen must be deterministic");
    }

    #[test]
    fn test_load_indexed_uses_lea_for_direct_storage() {
        let ir = IrProgram {
            string_data: vec![],
            global_data: vec![],
            global_vars: std::collections::HashMap::new(),
            global_vregs: std::collections::HashMap::new(),
            functions: vec![IrFunction {
                name: "_start".to_string(),
                return_type: JStarType::Int,
                param_vregs: vec![],
                param_count: 0,
                blocks: vec![BasicBlock {
                    label: "entry".to_string(),
                    instructions: vec![
                        IrInst::Alloca {
                            dest: 0,
                            size: 16,
                            ty: JStarType::Byte,
                        },
                        IrInst::LoadIndexed {
                            dest: 1,
                            base: 0,
                            index: IrValue::Imm(0),
                            ty: JStarType::Byte,
                        },
                    ],
                    terminator: Terminator::Return(Some(IrValue::Reg(1))),
                }],
                next_vreg: 2,
            }],
        };

        let mc = generate(&ir).unwrap();
        assert!(
            contains_rbp_disp32_insn(&mc.text, 0x8D, -24),
            "direct storage base should use lea [rbp+offset]"
        );
    }

    #[test]
    fn test_load_indexed_loads_pointer_value_from_slot() {
        let ir = IrProgram {
            string_data: vec![],
            global_data: vec![],
            global_vars: std::collections::HashMap::new(),
            global_vregs: std::collections::HashMap::new(),
            functions: vec![IrFunction {
                name: "_start".to_string(),
                return_type: JStarType::Int,
                param_vregs: vec![],
                param_count: 0,
                blocks: vec![BasicBlock {
                    label: "entry".to_string(),
                    instructions: vec![
                        IrInst::Alloca {
                            dest: 0,
                            size: 16,
                            ty: JStarType::Byte,
                        },
                        IrInst::AddressOf { dest: 1, src: 0 },
                        IrInst::LoadIndexed {
                            dest: 2,
                            base: 1,
                            index: IrValue::Imm(0),
                            ty: JStarType::Byte,
                        },
                    ],
                    terminator: Terminator::Return(Some(IrValue::Reg(2))),
                }],
                next_vreg: 3,
            }],
        };

        let mc = generate(&ir).unwrap();
        assert!(
            contains_rbp_disp32_insn(&mc.text, 0x8B, -32),
            "pointer-valued slot should be loaded before indexed access"
        );
        assert!(
            !contains_rbp_disp32_insn(&mc.text, 0x8D, -32),
            "pointer-valued slot must not be treated as direct storage"
        );
    }

    #[test]
    fn test_store_indexed_loads_pointer_value_from_slot() {
        let ir = IrProgram {
            string_data: vec![],
            global_data: vec![],
            global_vars: std::collections::HashMap::new(),
            global_vregs: std::collections::HashMap::new(),
            functions: vec![IrFunction {
                name: "_start".to_string(),
                return_type: JStarType::Int,
                param_vregs: vec![],
                param_count: 0,
                blocks: vec![BasicBlock {
                    label: "entry".to_string(),
                    instructions: vec![
                        IrInst::Alloca {
                            dest: 0,
                            size: 16,
                            ty: JStarType::Byte,
                        },
                        IrInst::AddressOf { dest: 1, src: 0 },
                        IrInst::StoreIndexed {
                            base: 1,
                            index: IrValue::Imm(0),
                            value: IrValue::Imm(7),
                            ty: JStarType::Byte,
                        },
                    ],
                    terminator: Terminator::Return(Some(IrValue::Imm(0))),
                }],
                next_vreg: 2,
            }],
        };

        let mc = generate(&ir).unwrap();
        assert!(
            contains_rbp_disp32_insn(&mc.text, 0x8B, -32),
            "indexed store should load pointer-valued base from its stack slot"
        );
        assert!(
            !contains_rbp_disp32_insn(&mc.text, 0x8D, -32),
            "indexed store must not treat pointer-valued base as direct storage"
        );
    }

    #[test]
    fn test_rex_prefix() {
        // r8-r15 need REX extension
        assert!(X86Reg::R8.needs_rex_ext());
        assert!(X86Reg::R15.needs_rex_ext());
        assert!(!X86Reg::Rax.needs_rex_ext());
        assert!(!X86Reg::Rdi.needs_rex_ext());
    }

    #[test]
    fn test_register_encoding() {
        assert_eq!(X86Reg::Rax.encoding(), 0);
        assert_eq!(X86Reg::Rcx.encoding(), 1);
        assert_eq!(X86Reg::R8.encoding(), 0); // lower 3 bits
        assert_eq!(X86Reg::R15.encoding(), 7);
    }
}
