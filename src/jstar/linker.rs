//! ELF Linker — Phase 6 of the JStar compiler.
//!
//! Assembles x86-64 machine code into a minimal ELF64 executable.
//! Static linking only in the bootstrap phase (no dynamic linking).
//!
//! Output format:
//!   ELF64 header (64 bytes)
//!   Program header table (1 entry = 56 bytes)
//!   .text section (executable code)
//!   .data section (if any)
//!
//! The _start entry point is at the beginning of .text.

use super::codegen::MachineCode;
use crate::types::{MorphResult, MorphlexError};
use std::path::Path;

// ─── ELF64 Constants ────────────────────────────────────────────────────────

// ELF magic
const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];

// ELF class
const ELFCLASS64: u8 = 2;

// ELF data encoding
const ELFDATA2LSB: u8 = 1; // little-endian

// ELF version
const EV_CURRENT: u8 = 1;

// ELF OS/ABI
const ELFOSABI_NONE: u8 = 0; // System V

// ELF type
const ET_EXEC: u16 = 2; // executable

// ELF machine
const EM_X86_64: u16 = 62;

// Program header types
const PT_LOAD: u32 = 1;

// Program header flags
const PF_X: u32 = 1; // execute
const PF_W: u32 = 2; // write
const PF_R: u32 = 4; // read

// Header sizes
const ELF64_EHDR_SIZE: usize = 64;
const ELF64_PHDR_SIZE: usize = 56;

// Virtual address base (standard Linux user-space)
const VADDR_BASE: u64 = 0x400000;

// Kernel virtual address base (conventional 1 MB mark)
const KERNEL_VADDR_BASE: u64 = 0x100000;

// Multiboot2 constants
const MULTIBOOT2_MAGIC: u32 = 0xE85250D6;
const MULTIBOOT2_ARCH_X86: u32 = 0;

/// Link machine code into an ELF64 executable.
pub fn link(code: &MachineCode, output_path: &Path) -> MorphResult<()> {
    // Patch data section addresses in the .text before building ELF
    let mut code = code.clone();
    patch_data_addresses(&mut code);
    let elf = build_elf(&code)?;

    std::fs::write(output_path, &elf).map_err(|e| MorphlexError::IoError(e))?;

    // Set executable permission
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(output_path, perms).map_err(|e| MorphlexError::IoError(e))?;
    }

    Ok(())
}

/// Link machine code into a Multiboot2-bootable ELF64 kernel image.
///
/// Produces a bare-metal kernel loadable by QEMU `-kernel` or any
/// Multiboot2-compliant bootloader. The binary starts with a 32-bit
/// protected-mode stub that:
///   1. Sets up identity-mapped page tables (first 2 MB via 2 MB page)
///   2. Enables PAE (CR4.PAE)
///   3. Loads page tables into CR3
///   4. Sets IA32_EFER.LME (Long Mode Enable)
///   5. Enables paging (CR0.PG)
///   6. Far-jumps into 64-bit mode
///   7. Falls through to the JStar kernel code
pub fn link_kernel(code: &MachineCode, output_path: &Path) -> MorphResult<()> {
    let mut code = code.clone();
    let stub = build_boot_stub(code.text.len());
    // Prepend stub; adjust data fixups to account for stub offset
    for fixup in &mut code.data_fixups {
        *fixup += stub.len();
    }
    let mut full_text = stub;
    full_text.extend_from_slice(&code.text);
    code.text = full_text;

    patch_kernel_data_addresses(&mut code);
    let elf = build_kernel_elf(&code)?;

    std::fs::write(output_path, &elf).map_err(|e| MorphlexError::IoError(e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(output_path, perms).map_err(|e| MorphlexError::IoError(e))?;
    }

    Ok(())
}

/// Build the 32-bit → 64-bit boot stub with Multiboot2 header.
///
/// Layout:
///   [Multiboot2 header: 24 bytes (header + end tag)]
///   [32-bit code: ~120 bytes]
///   [GDT: 24 bytes (null + code64 + data64)]
///   [GDT pointer: 6 bytes]
///   [Page tables: 3 * 4096 = 12288 bytes (PML4 + PDPT + PD)]
///
/// Total stub ≈ 12,464 bytes, page-table aligned to 4096.
fn build_boot_stub(_kernel_code_len: usize) -> Vec<u8> {
    let mut stub = Vec::new();

    // ─── Multiboot2 Header ─────────────────────────────────────────────
    // Must appear in the first 32768 bytes of the binary, 8-byte aligned.
    let mb2_header_len: u32 = 16; // magic(4) + arch(4) + len(4) + checksum(4) + end_tag(8)
    let mb2_total = mb2_header_len + 8; // include end tag
    let checksum = (0u32.wrapping_sub(MULTIBOOT2_MAGIC.wrapping_add(MULTIBOOT2_ARCH_X86).wrapping_add(mb2_total))) as u32;

    stub.extend_from_slice(&MULTIBOOT2_MAGIC.to_le_bytes());
    stub.extend_from_slice(&MULTIBOOT2_ARCH_X86.to_le_bytes());
    stub.extend_from_slice(&mb2_total.to_le_bytes());
    stub.extend_from_slice(&checksum.to_le_bytes());
    // End tag (type=0, flags=0, size=8)
    stub.extend_from_slice(&0u16.to_le_bytes()); // type
    stub.extend_from_slice(&0u16.to_le_bytes()); // flags
    stub.extend_from_slice(&8u32.to_le_bytes()); // size
    let _mb2_end = stub.len();

    // ─── 32-bit Protected Mode Stub ────────────────────────────────────
    // Entered by Multiboot2 loader at this point. CPU is in 32-bit
    // protected mode, paging disabled, A20 enabled.
    //
    // We need to:
    //   1. Load page tables (identity map first 2 MB)
    //   2. Enable PAE in CR4
    //   3. Set IA32_EFER.LME
    //   4. Load CR3 with PML4 address
    //   5. Enable paging in CR0
    //   6. Load 64-bit GDT
    //   7. Far jump to 64-bit code

    // We'll calculate addresses relative to the load address.
    // The stub is at the beginning of .text, which starts after ELF headers.
    let headers_size = ELF64_EHDR_SIZE + ELF64_PHDR_SIZE; // 120 bytes
    let stub_base = KERNEL_VADDR_BASE + headers_size as u64;

    // Page tables will be at a 4096-aligned offset within the stub.
    // We'll place: [mb2 header: 24][32-bit code: variable][GDT: 24][GDT ptr: 10][padding][page tables: 12288]
    // First, emit the 32-bit code, then we'll know offsets.

    let _code32_start = stub.len();

    // All 32-bit code is emitted as raw bytes. We use 32-bit operand/address sizes.

    // Disable interrupts: cli
    stub.push(0xFA);

    // We'll fill in page table and GDT addresses after laying out the stub.
    // For now, emit placeholder instructions and patch later.

    // Save stub code position for patching
    // Step 1: Load PML4 address into CR3
    //   mov eax, <pml4_addr>    ; B8 xx xx xx xx
    //   mov cr3, eax            ; 0F 22 D8
    let mov_eax_pml4_pos = stub.len();
    stub.extend_from_slice(&[0xB8, 0x00, 0x00, 0x00, 0x00]); // patched later
    stub.extend_from_slice(&[0x0F, 0x22, 0xD8]); // mov cr3, eax

    // Step 2: Enable PAE in CR4
    //   mov eax, cr4            ; 0F 20 E0
    //   or eax, 0x20            ; 83 C8 20
    //   mov cr4, eax            ; 0F 22 E0
    stub.extend_from_slice(&[0x0F, 0x20, 0xE0]); // mov eax, cr4
    stub.extend_from_slice(&[0x83, 0xC8, 0x20]); // or eax, 0x20 (PAE bit)
    stub.extend_from_slice(&[0x0F, 0x22, 0xE0]); // mov cr4, eax

    // Step 3: Set IA32_EFER.LME (bit 8)
    //   mov ecx, 0xC0000080     ; B9 80 00 00 C0
    //   rdmsr                    ; 0F 32
    //   or eax, 0x100            ; 0D 00 01 00 00
    //   wrmsr                    ; 0F 30
    stub.extend_from_slice(&[0xB9, 0x80, 0x00, 0x00, 0xC0]); // mov ecx, IA32_EFER
    stub.extend_from_slice(&[0x0F, 0x32]); // rdmsr
    stub.extend_from_slice(&[0x0D, 0x00, 0x01, 0x00, 0x00]); // or eax, 0x100
    stub.extend_from_slice(&[0x0F, 0x30]); // wrmsr

    // Step 4: Enable paging + protected mode in CR0
    //   mov eax, cr0            ; 0F 20 C0
    //   or eax, 0x80000001      ; 0D 01 00 00 80
    //   mov cr0, eax            ; 0F 22 C0
    stub.extend_from_slice(&[0x0F, 0x20, 0xC0]); // mov eax, cr0
    stub.extend_from_slice(&[0x0D, 0x01, 0x00, 0x00, 0x80]); // or eax, PG|PE
    stub.extend_from_slice(&[0x0F, 0x22, 0xC0]); // mov cr0, eax

    // Step 5: Load 64-bit GDT
    //   lgdt [gdt_ptr]          ; 0F 01 15 xx xx xx xx
    let lgdt_addr_pos = stub.len() + 3;
    stub.extend_from_slice(&[0x0F, 0x01, 0x15, 0x00, 0x00, 0x00, 0x00]); // patched later

    // Step 6: Far jump to 64-bit code
    //   jmp 0x08:<kernel64_entry>  ; EA xx xx xx xx 08 00
    let jmp_addr_pos = stub.len() + 1;
    stub.extend_from_slice(&[0xEA, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00]); // patched later

    // ─── 64-bit entry point ────────────────────────────────────────────
    // After the far jump, we land here in 64-bit mode.
    let entry64_offset = stub.len();

    // Set up data segment registers (selector 0x10 = second GDT entry = data64)
    //   mov ax, 0x10            ; 66 B8 10 00
    //   mov ds, ax              ; 8E D8
    //   mov es, ax              ; 8E C0
    //   mov ss, ax              ; 8E D0
    //   xor ax, ax              ; 66 31 C0
    //   mov fs, ax              ; 8E E0
    //   mov gs, ax              ; 8E E8
    stub.extend_from_slice(&[0x66, 0xB8, 0x10, 0x00]); // mov ax, 0x10
    stub.extend_from_slice(&[0x8E, 0xD8]); // mov ds, ax
    stub.extend_from_slice(&[0x8E, 0xC0]); // mov es, ax
    stub.extend_from_slice(&[0x8E, 0xD0]); // mov ss, ax
    stub.extend_from_slice(&[0x66, 0x31, 0xC0]); // xor ax, ax
    stub.extend_from_slice(&[0x8E, 0xE0]); // mov fs, ax
    stub.extend_from_slice(&[0x8E, 0xE8]); // mov gs, ax

    // Set up a kernel stack at the end of the first 2 MB identity map
    //   mov rsp, 0x200000 - 16   ; 48 BC xx xx xx xx xx xx xx xx
    let stack_top: u64 = 0x200000 - 16;
    stub.push(0x48); stub.push(0xBC); // movabs rsp, imm64
    stub.extend_from_slice(&stack_top.to_le_bytes());

    // The JStar kernel code starts right after this stub.
    // We just fall through — the next bytes ARE the kernel.
    // No jump needed since kernel code is contiguous.

    let _code32_end = stub.len();

    // ─── GDT (Global Descriptor Table) ─────────────────────────────────
    // 3 entries: null, code64, data64
    let gdt_offset = stub.len();

    // Null descriptor (8 bytes)
    stub.extend_from_slice(&[0x00; 8]);

    // Code64 descriptor: base=0, limit=0xFFFFF, type=Execute/Read, L=1 (64-bit), G=1
    // Bytes: limit_low=0xFFFF, base_low=0x0000, base_mid=0x00,
    //        access=0x9A (P=1,DPL=0,S=1,E=1,RW=1), flags_limit=0xAF (G=1,L=1,limit_hi=0xF),
    //        base_high=0x00
    stub.extend_from_slice(&[0xFF, 0xFF, 0x00, 0x00, 0x00, 0x9A, 0xAF, 0x00]);

    // Data64 descriptor: base=0, limit=0xFFFFF, type=Read/Write
    // access=0x92 (P=1,DPL=0,S=1,W=1), flags=0xCF (G=1,DB=1)
    stub.extend_from_slice(&[0xFF, 0xFF, 0x00, 0x00, 0x00, 0x92, 0xCF, 0x00]);

    // ─── GDT Pointer (6 bytes: limit u16 + base u32) ──────────────────
    let gdt_ptr_offset = stub.len();
    let gdt_limit: u16 = 3 * 8 - 1; // 23
    stub.extend_from_slice(&gdt_limit.to_le_bytes());
    // GDT base address (32-bit, will be patched)
    let gdt_base_patch_pos = stub.len();
    stub.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // patched later

    // Align to 4096 for page tables
    while stub.len() % 4096 != 0 {
        stub.push(0x00);
    }

    // ─── Page Tables (identity map first 2 MB) ─────────────────────────
    // PML4 → PDPT → PD (using 2 MB pages, so no PT needed)
    let pml4_offset = stub.len();
    let pdpt_offset = pml4_offset + 4096;
    let pd_offset = pdpt_offset + 4096;

    // PML4: entry 0 points to PDPT
    let mut pml4 = vec![0u8; 4096];
    let pdpt_phys = stub_base + pdpt_offset as u64;
    let pdpt_entry = pdpt_phys | 0x03; // Present + Writable
    pml4[0..8].copy_from_slice(&pdpt_entry.to_le_bytes());
    stub.extend_from_slice(&pml4);

    // PDPT: entry 0 points to PD
    let mut pdpt = vec![0u8; 4096];
    let pd_phys = stub_base + pd_offset as u64;
    let pd_entry = pd_phys | 0x03; // Present + Writable
    pdpt[0..8].copy_from_slice(&pd_entry.to_le_bytes());
    stub.extend_from_slice(&pdpt);

    // PD: entry 0 = 2MB page at 0x000000 (identity map)
    let mut pd = vec![0u8; 4096];
    let page_entry: u64 = 0x00 | 0x83; // Present + Writable + PS (2MB page)
    pd[0..8].copy_from_slice(&page_entry.to_le_bytes());
    // Entry 1 = 2MB page at 0x200000 (covers VGA, kernel at 0x100000, etc.)
    // Actually entry 0 covers 0x000000-0x1FFFFF which includes 0x100000
    // But we also need 0x200000 range for stack
    let page_entry_1: u64 = 0x200000 | 0x83;
    pd[8..16].copy_from_slice(&page_entry_1.to_le_bytes());
    stub.extend_from_slice(&pd);

    // ─── Patch addresses ───────────────────────────────────────────────

    // PML4 physical address for CR3
    let pml4_addr = (stub_base + pml4_offset as u64) as u32;
    stub[mov_eax_pml4_pos + 1..mov_eax_pml4_pos + 5].copy_from_slice(&pml4_addr.to_le_bytes());

    // GDT pointer address for lgdt
    let gdt_ptr_addr = (stub_base + gdt_ptr_offset as u64) as u32;
    stub[lgdt_addr_pos..lgdt_addr_pos + 4].copy_from_slice(&gdt_ptr_addr.to_le_bytes());

    // GDT base in the GDT pointer
    let gdt_addr = (stub_base + gdt_offset as u64) as u32;
    stub[gdt_base_patch_pos..gdt_base_patch_pos + 4].copy_from_slice(&gdt_addr.to_le_bytes());

    // Far jump target: 64-bit entry point
    let entry64_addr = (stub_base + entry64_offset as u64) as u32;
    stub[jmp_addr_pos..jmp_addr_pos + 4].copy_from_slice(&entry64_addr.to_le_bytes());

    stub
}

/// Patch data section addresses for kernel image (uses KERNEL_VADDR_BASE).
fn patch_kernel_data_addresses(code: &mut MachineCode) {
    if code.data.is_empty() && code.data_fixups.is_empty() {
        return;
    }

    let headers_size = ELF64_EHDR_SIZE + ELF64_PHDR_SIZE;
    let data_offset = headers_size + code.text.len();
    let data_vaddr = KERNEL_VADDR_BASE + data_offset as u64;

    for &fixup_pos in &code.data_fixups {
        if fixup_pos + 8 <= code.text.len() {
            let offset_bytes: [u8; 8] = code.text[fixup_pos..fixup_pos + 8].try_into().unwrap();
            let current_val = u64::from_le_bytes(offset_bytes);
            let patched = current_val + data_vaddr;
            code.text[fixup_pos..fixup_pos + 8].copy_from_slice(&patched.to_le_bytes());
        }
    }
}

/// Build a Multiboot2-compatible ELF64 kernel image.
fn build_kernel_elf(code: &MachineCode) -> MorphResult<Vec<u8>> {
    let text_size = code.text.len();
    let data_size = code.data.len();
    let headers_size = ELF64_EHDR_SIZE + ELF64_PHDR_SIZE;
    let segment_size = text_size + data_size;
    let mem_segment_size = segment_size + code.bss_size;

    // Entry point is the start of .text (32-bit stub)
    let entry_point = KERNEL_VADDR_BASE + headers_size as u64;

    let mut elf = Vec::with_capacity(headers_size + segment_size);

    // ─── ELF Header ────────────────────────────────────────────────────
    elf.extend_from_slice(&ELF_MAGIC);
    elf.push(ELFCLASS64);
    elf.push(ELFDATA2LSB);
    elf.push(EV_CURRENT);
    elf.push(ELFOSABI_NONE);
    elf.extend_from_slice(&[0u8; 8]);

    elf.extend_from_slice(&ET_EXEC.to_le_bytes());
    elf.extend_from_slice(&EM_X86_64.to_le_bytes());
    elf.extend_from_slice(&1u32.to_le_bytes());
    elf.extend_from_slice(&entry_point.to_le_bytes());
    elf.extend_from_slice(&(ELF64_EHDR_SIZE as u64).to_le_bytes());
    elf.extend_from_slice(&0u64.to_le_bytes()); // shoff
    elf.extend_from_slice(&0u32.to_le_bytes()); // flags
    elf.extend_from_slice(&(ELF64_EHDR_SIZE as u16).to_le_bytes());
    elf.extend_from_slice(&(ELF64_PHDR_SIZE as u16).to_le_bytes());
    elf.extend_from_slice(&1u16.to_le_bytes()); // phnum
    elf.extend_from_slice(&0u16.to_le_bytes());
    elf.extend_from_slice(&0u16.to_le_bytes());
    elf.extend_from_slice(&0u16.to_le_bytes());

    assert_eq!(elf.len(), ELF64_EHDR_SIZE);

    // ─── PT_LOAD (R+W+X) ──────────────────────────────────────────────
    elf.extend_from_slice(&PT_LOAD.to_le_bytes());
    elf.extend_from_slice(&(PF_R | PF_W | PF_X).to_le_bytes());
    elf.extend_from_slice(&0u64.to_le_bytes()); // p_offset
    elf.extend_from_slice(&KERNEL_VADDR_BASE.to_le_bytes());
    elf.extend_from_slice(&KERNEL_VADDR_BASE.to_le_bytes());
    let total_file_size = (headers_size + segment_size) as u64;
    let total_mem_size = (headers_size + mem_segment_size) as u64;
    elf.extend_from_slice(&total_file_size.to_le_bytes());
    elf.extend_from_slice(&total_mem_size.to_le_bytes());
    elf.extend_from_slice(&0x1000u64.to_le_bytes());

    assert_eq!(elf.len(), headers_size);

    // ─── .text (includes boot stub + kernel code) ──────────────────────
    elf.extend_from_slice(&code.text);

    // ─── .data ─────────────────────────────────────────────────────────
    if data_size > 0 {
        elf.extend_from_slice(&code.data);
    }

    Ok(elf)
}

/// Patch data section addresses in the .text section.
///
/// Uses the data_fixups list from codegen: each entry is the byte offset
/// in .text of an 8-byte value (a .data section offset) to which we add
/// the actual data vaddr (VADDR_BASE + headers + text_size).
///
/// This replaces the old byte-pattern scanning approach. Every movabs
/// that references .data now records its fixup position explicitly.
fn patch_data_addresses(code: &mut MachineCode) {
    if code.data.is_empty() && code.data_fixups.is_empty() {
        return;
    }

    let headers_size = ELF64_EHDR_SIZE + ELF64_PHDR_SIZE;
    let data_offset = headers_size + code.text.len();
    let data_vaddr = VADDR_BASE + data_offset as u64;

    for &fixup_pos in &code.data_fixups {
        if fixup_pos + 8 <= code.text.len() {
            let offset_bytes: [u8; 8] = code.text[fixup_pos..fixup_pos + 8].try_into().unwrap();
            let current_val = u64::from_le_bytes(offset_bytes);
            let patched = current_val + data_vaddr;
            code.text[fixup_pos..fixup_pos + 8].copy_from_slice(&patched.to_le_bytes());
        }
    }
}

/// Build the complete ELF64 binary in memory.
///
/// Uses a single PT_LOAD segment (R+W+X) for the bootstrap compiler.
/// This avoids multi-segment mapping complexity. All code and data
/// are in one segment mapped at VADDR_BASE.
fn build_elf(code: &MachineCode) -> MorphResult<Vec<u8>> {
    let text_size = code.text.len();
    let data_size = code.data.len();

    let headers_size = ELF64_EHDR_SIZE + ELF64_PHDR_SIZE;

    // File segment = text + initialized data (no BSS)
    let segment_size = text_size + data_size;
    // Memory segment = file segment + BSS (zero-filled by kernel)
    let mem_segment_size = segment_size + code.bss_size;

    // Entry point = start of .text (right after headers)
    let entry_point = VADDR_BASE + headers_size as u64;

    let mut elf = Vec::with_capacity(headers_size + segment_size);

    // ─── ELF Header (64 bytes) ──────────────────────────────────────────

    elf.extend_from_slice(&ELF_MAGIC);
    elf.push(ELFCLASS64);
    elf.push(ELFDATA2LSB);
    elf.push(EV_CURRENT);
    elf.push(ELFOSABI_NONE);
    elf.extend_from_slice(&[0u8; 8]); // padding

    elf.extend_from_slice(&ET_EXEC.to_le_bytes());
    elf.extend_from_slice(&EM_X86_64.to_le_bytes());
    elf.extend_from_slice(&1u32.to_le_bytes()); // version
    elf.extend_from_slice(&entry_point.to_le_bytes());
    elf.extend_from_slice(&(ELF64_EHDR_SIZE as u64).to_le_bytes()); // phoff
    elf.extend_from_slice(&0u64.to_le_bytes()); // shoff
    elf.extend_from_slice(&0u32.to_le_bytes()); // flags
    elf.extend_from_slice(&(ELF64_EHDR_SIZE as u16).to_le_bytes());
    elf.extend_from_slice(&(ELF64_PHDR_SIZE as u16).to_le_bytes());
    elf.extend_from_slice(&1u16.to_le_bytes()); // phnum = 1
    elf.extend_from_slice(&0u16.to_le_bytes()); // shentsize
    elf.extend_from_slice(&0u16.to_le_bytes()); // shnum
    elf.extend_from_slice(&0u16.to_le_bytes()); // shstrndx

    assert_eq!(elf.len(), ELF64_EHDR_SIZE);

    // ─── Single Program Header: PT_LOAD (R+W+X) ────────────────────────
    // Maps the entire file from offset 0 so header/text/data are all in one segment.

    elf.extend_from_slice(&PT_LOAD.to_le_bytes());
    elf.extend_from_slice(&(PF_R | PF_W | PF_X).to_le_bytes()); // rwx
    elf.extend_from_slice(&0u64.to_le_bytes()); // p_offset: start of file
    elf.extend_from_slice(&VADDR_BASE.to_le_bytes()); // p_vaddr
    elf.extend_from_slice(&VADDR_BASE.to_le_bytes()); // p_paddr
    let total_file_size = (headers_size + segment_size) as u64;
    let total_mem_size = (headers_size + mem_segment_size) as u64;
    elf.extend_from_slice(&total_file_size.to_le_bytes()); // p_filesz
    elf.extend_from_slice(&total_mem_size.to_le_bytes()); // p_memsz (includes BSS)
    elf.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align

    assert_eq!(elf.len(), headers_size);

    // ─── .text section ──────────────────────────────────────────────────

    elf.extend_from_slice(&code.text);

    // ─── .data section ──────────────────────────────────────────────────

    if data_size > 0 {
        elf.extend_from_slice(&code.data);
    }

    Ok(elf)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elf_magic() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0x90], // nop
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_elf(&code).unwrap();
        assert_eq!(&elf[0..4], &ELF_MAGIC);
    }

    #[test]
    fn test_elf_class_64() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0x90],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_elf(&code).unwrap();
        assert_eq!(elf[4], ELFCLASS64);
    }

    #[test]
    fn test_elf_machine_x86_64() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0x90],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_elf(&code).unwrap();
        let machine = u16::from_le_bytes([elf[18], elf[19]]);
        assert_eq!(machine, EM_X86_64);
    }

    #[test]
    fn test_elf_header_size() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0x90],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_elf(&code).unwrap();
        // ELF header (64) + 1 phdr (56) + 1 byte text = 121
        assert_eq!(elf.len(), ELF64_EHDR_SIZE + ELF64_PHDR_SIZE + 1);
    }

    #[test]
    fn test_elf_entry_point() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0x90],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_elf(&code).unwrap();
        let entry = u64::from_le_bytes(elf[24..32].try_into().unwrap());
        let expected = VADDR_BASE + (ELF64_EHDR_SIZE + ELF64_PHDR_SIZE) as u64;
        assert_eq!(entry, expected);
    }

    #[test]
    fn test_elf_with_data_section() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0x90],
            data: vec![0x42, 0x43],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_elf(&code).unwrap();
        // Single PT_LOAD segment — always 1 program header
        let phnum = u16::from_le_bytes([elf[56], elf[57]]);
        assert_eq!(phnum, 1);
        // Total size: header + 1 phdr + 1 text + 2 data
        assert_eq!(elf.len(), ELF64_EHDR_SIZE + ELF64_PHDR_SIZE + 1 + 2);
    }

    #[test]
    fn test_elf_determinism() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0xB8, 0x01, 0x00, 0x00, 0x00], // mov eax, 1
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let a = build_elf(&code).unwrap();
        let b = build_elf(&code).unwrap();
        assert_eq!(a, b, "ELF output must be deterministic");
    }

    #[test]
    fn test_kernel_multiboot2_magic() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0xF4], // hlt
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_kernel_elf_from_code(&code);
        let text_start = ELF64_EHDR_SIZE + ELF64_PHDR_SIZE;
        let magic = u32::from_le_bytes(elf[text_start..text_start + 4].try_into().unwrap());
        assert_eq!(magic, MULTIBOOT2_MAGIC);
    }

    #[test]
    fn test_kernel_multiboot2_checksum() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0xF4],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_kernel_elf_from_code(&code);
        let t = ELF64_EHDR_SIZE + ELF64_PHDR_SIZE;
        let magic = u32::from_le_bytes(elf[t..t+4].try_into().unwrap());
        let arch = u32::from_le_bytes(elf[t+4..t+8].try_into().unwrap());
        let len = u32::from_le_bytes(elf[t+8..t+12].try_into().unwrap());
        let checksum = u32::from_le_bytes(elf[t+12..t+16].try_into().unwrap());
        assert_eq!(magic.wrapping_add(arch).wrapping_add(len).wrapping_add(checksum), 0);
    }

    #[test]
    fn test_kernel_entry_at_0x100000() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0xF4],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_kernel_elf_from_code(&code);
        let entry = u64::from_le_bytes(elf[24..32].try_into().unwrap());
        let expected = KERNEL_VADDR_BASE + (ELF64_EHDR_SIZE + ELF64_PHDR_SIZE) as u64;
        assert_eq!(entry, expected);
    }

    #[test]
    fn test_kernel_vaddr_base() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0xF4],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_kernel_elf_from_code(&code);
        // p_vaddr is at offset 64 (ehdr) + 16 (p_type + p_flags + p_offset) = 80
        let p_vaddr = u64::from_le_bytes(elf[80..88].try_into().unwrap());
        assert_eq!(p_vaddr, KERNEL_VADDR_BASE);
    }

    #[test]
    fn test_kernel_determinism() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0xF4],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let a = build_kernel_elf_from_code(&code);
        let b = build_kernel_elf_from_code(&code);
        assert_eq!(a, b, "Kernel ELF output must be deterministic");
    }

    #[test]
    fn test_kernel_boot_stub_has_cli() {
        let code = MachineCode {
            data_vaddr: 0,
            text: vec![0xF4],
            data: vec![],
            bss_size: 0,
            stack_size: 0,
            data_fixups: vec![],
        };
        let elf = build_kernel_elf_from_code(&code);
        let text_start = ELF64_EHDR_SIZE + ELF64_PHDR_SIZE;
        // After 24-byte Multiboot2 header, first instruction should be CLI (0xFA)
        assert_eq!(elf[text_start + 24], 0xFA);
    }

    fn build_kernel_elf_from_code(code: &MachineCode) -> Vec<u8> {
        let mut code = code.clone();
        let stub = build_boot_stub(code.text.len());
        for fixup in &mut code.data_fixups {
            *fixup += stub.len();
        }
        let mut full_text = stub;
        full_text.extend_from_slice(&code.text);
        code.text = full_text;
        patch_kernel_data_addresses(&mut code);
        build_kernel_elf(&code).unwrap()
    }
}
