//! IR Optimization Pass -- runs between IR lowering and codegen.
//!
//! Three passes, applied per-function:
//!   1. Constant Folding  -- evaluate BinOps on two immediates at compile time
//!   2. Copy Propagation  -- propagate Imm values through Copy instructions
//!   3. Dead Code Elimination -- remove Nop and unused instructions

use std::collections::HashSet;
use super::ir::*;

/// Run all optimization passes on the IR program (mutates in place).
pub fn optimize(program: &mut IrProgram) {
    for func in &mut program.functions {
        optimize_function(func);
    }
}

/// Run all optimization passes on a single function.
pub fn optimize_function(func: &mut IrFunction) {
    for block in &mut func.blocks {
        copy_propagate(block);
        constant_fold(block);
        copy_propagate(block);
    }
    dead_code_eliminate(func);
}

fn constant_fold(block: &mut BasicBlock) {
    for inst in &mut block.instructions {
        let folded = match inst {
            IrInst::BinOp { dest, op, lhs: IrValue::Imm(l), rhs: IrValue::Imm(r), ty } => {
                let l = *l;
                let r = *r;
                let result = match op {
                    IrBinOp::Add => Some(l.wrapping_add(r)),
                    IrBinOp::Sub => Some(l.wrapping_sub(r)),
                    IrBinOp::Mul => Some(l.wrapping_mul(r)),
                    IrBinOp::Div => match r {
                        0 => None,
                        -1 if l == i64::MIN => None,
                        _ => Some(l.wrapping_div(r)),
                    },
                    IrBinOp::Mod => match r {
                        0 => None,
                        -1 if l == i64::MIN => None,
                        _ => Some(l.wrapping_rem(r)),
                    },
                    IrBinOp::And => Some(l & r),
                    IrBinOp::Or  => Some(l | r),
                    IrBinOp::Xor => Some(l ^ r),
                    IrBinOp::Shl => Some(l.wrapping_shl(r as u32)),
                    IrBinOp::Shr => Some(l.wrapping_shr(r as u32)),
                };
                result.map(|val| (*dest, val, *ty))
            }
            _ => None,
        };
        if let Some((dest, val, ty)) = folded {
            *inst = IrInst::Copy {
                dest,
                src: IrValue::Imm(val),
                ty,
            };
        }
    }
}

fn copy_propagate(block: &mut BasicBlock) {
    use std::collections::HashMap;
    let mut imm_map: HashMap<VReg, i64> = HashMap::new();
    for inst in &mut block.instructions {
        replace_values_in_inst(inst, &imm_map);

        if let Some(dest) = inst_dest(inst) {
            imm_map.remove(&dest);
        }

        if let IrInst::Copy { dest, src: IrValue::Imm(n), .. } = inst {
            imm_map.insert(*dest, *n);
        }
    }
    replace_values_in_terminator(&mut block.terminator, &imm_map);
}

fn replace_values_in_inst(inst: &mut IrInst, map: &std::collections::HashMap<VReg, i64>) {
    if map.is_empty() { return; }
    match inst {
        IrInst::BinOp { lhs, rhs, .. } => { replace_value(lhs, map); replace_value(rhs, map); }
        IrInst::UnaryOp { src, .. } => { replace_value(src, map); }
        IrInst::Copy { src, .. } => { replace_value(src, map); }
        IrInst::Store { addr, value, .. } => { replace_value(addr, map); replace_value(value, map); }
        IrInst::Load { addr, .. } => { replace_value(addr, map); }
        IrInst::Call { args, .. } => { for arg in args { replace_value(arg, map); } }
        IrInst::Syscall { number, args, .. } => { replace_value(number, map); for arg in args { replace_value(arg, map); } }
        IrInst::Compare { lhs, rhs, .. } => { replace_value(lhs, map); replace_value(rhs, map); }
        IrInst::Print { value } => { replace_value(value, map); }
        IrInst::ArrayLoad { base, index, .. } => { replace_value(base, map); replace_value(index, map); }
        IrInst::ArrayStore { base, index, value } => { replace_value(base, map); replace_value(index, map); replace_value(value, map); }
        IrInst::StoreIndexed { index, value, .. } => { replace_value(index, map); replace_value(value, map); }
        IrInst::LoadIndexed { index, .. } => { replace_value(index, map); }
        IrInst::HashOp { addr, len, .. } => { replace_value(addr, map); replace_value(len, map); }
        IrInst::FileRead { fd, buf, len, .. } => { replace_value(fd, map); replace_value(buf, map); replace_value(len, map); }
        IrInst::FileClose { fd } => { replace_value(fd, map); }
        IrInst::StrCmp { a, b, len, .. } => { replace_value(a, map); replace_value(b, map); replace_value(len, map); }
        IrInst::StrLen { addr, .. } => { replace_value(addr, map); }
        IrInst::StrCopy { dst, src, len } => { replace_value(dst, map); replace_value(src, map); replace_value(len, map); }
        IrInst::Outb { port, value } => { replace_value(port, map); replace_value(value, map); }
        IrInst::Inb { port, .. } => { replace_value(port, map); }
        IrInst::Lgdt { addr } | IrInst::Lidt { addr } | IrInst::Invlpg { addr } => { replace_value(addr, map); }
        IrInst::Wrmsr { msr, value } => { replace_value(msr, map); replace_value(value, map); }
        IrInst::Rdmsr { msr, .. } => { replace_value(msr, map); }
        IrInst::WriteCr { cr, value } => { replace_value(cr, map); replace_value(value, map); }
        IrInst::ReadCr { cr, .. } => { replace_value(cr, map); }
        IrInst::Ltr { selector } => { replace_value(selector, map); }
        IrInst::StoreAbs { addr, value, .. } => { replace_value(addr, map); replace_value(value, map); }
        IrInst::LoadAbs { addr, .. } => { replace_value(addr, map); }
        IrInst::AddressOf { .. } | IrInst::Alloca { .. } | IrInst::PrintStr { .. } | IrInst::ArrayAlloc { .. } | IrInst::FileOpen { .. } | IrInst::ArrayLength { .. } | IrInst::Nop | IrInst::Cli | IrInst::Sti | IrInst::Iretq => {}
    }
}

fn replace_values_in_terminator(term: &mut Terminator, map: &std::collections::HashMap<VReg, i64>) {
    match term {
        Terminator::Return(Some(val)) => { replace_value(val, map); }
        Terminator::Halt(val) => { replace_value(val, map); }
        Terminator::Return(None) | Terminator::Jump(_) | Terminator::Branch { .. } | Terminator::Unreachable => {}
    }
}

fn replace_value(val: &mut IrValue, map: &std::collections::HashMap<VReg, i64>) {
    match val {
        IrValue::Reg(r) => { if let Some(&imm) = map.get(r) { *val = IrValue::Imm(imm); } }
        _ => {}
    }
}

fn dead_code_eliminate(func: &mut IrFunction) {
    loop {
        let used = collect_used_vregs(func);
        let mut changed = false;

        for block in &mut func.blocks {
            let before_len = block.instructions.len();
            block.instructions.retain(|inst| {
                match inst {
                    IrInst::Nop => false,
                    IrInst::Store { .. }
                    | IrInst::StoreIndexed { .. }
                    | IrInst::Call { .. }
                    | IrInst::Syscall { .. }
                    | IrInst::Print { .. }
                    | IrInst::PrintStr { .. }
                    | IrInst::ArrayStore { .. }
                    | IrInst::FileOpen { .. }
                    | IrInst::FileRead { .. }
                    | IrInst::FileClose { .. }
                    | IrInst::StrCopy { .. } => true,
                    _ => match inst_dest(inst) {
                        Some(dest) => used.contains(&dest),
                        None => true,
                    },
                }
            });

            if block.instructions.len() != before_len {
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

fn inst_dest(inst: &IrInst) -> Option<VReg> {
    match inst {
        IrInst::BinOp { dest, .. } | IrInst::UnaryOp { dest, .. } | IrInst::Copy { dest, .. } | IrInst::Load { dest, .. } | IrInst::AddressOf { dest, .. } | IrInst::Call { dest, .. } | IrInst::Syscall { dest, .. } | IrInst::Alloca { dest, .. } | IrInst::Compare { dest, .. } | IrInst::ArrayAlloc { dest, .. } | IrInst::ArrayLoad { dest, .. } | IrInst::LoadIndexed { dest, .. } | IrInst::HashOp { dest, .. } | IrInst::FileOpen { dest, .. } | IrInst::FileRead { dest, .. } | IrInst::ArrayLength { dest, .. } | IrInst::StrCmp { dest, .. } | IrInst::StrLen { dest, .. } => Some(*dest),
        IrInst::Inb { dest, .. } | IrInst::Rdmsr { dest, .. } | IrInst::ReadCr { dest, .. } | IrInst::LoadAbs { dest, .. } => Some(*dest),
        IrInst::Store { .. } | IrInst::StoreIndexed { .. } | IrInst::Print { .. } | IrInst::PrintStr { .. } | IrInst::ArrayStore { .. } | IrInst::FileClose { .. } | IrInst::StrCopy { .. } | IrInst::Nop | IrInst::Outb { .. } | IrInst::Cli | IrInst::Sti | IrInst::Lgdt { .. } | IrInst::Lidt { .. } | IrInst::Wrmsr { .. } | IrInst::WriteCr { .. } | IrInst::Iretq | IrInst::Ltr { .. } | IrInst::Invlpg { .. } | IrInst::StoreAbs { .. } => None,
    }
}

fn collect_used_vregs(func: &IrFunction) -> HashSet<VReg> {
    let mut used = HashSet::new();
    for block in &func.blocks {
        for inst in &block.instructions { collect_values_in_inst(inst, &mut used); }
        collect_values_in_terminator(&block.terminator, &mut used);
    }
    used
}

fn collect_values_in_inst(inst: &IrInst, used: &mut HashSet<VReg>) {
    match inst {
        IrInst::BinOp { lhs, rhs, .. } => { collect_vreg(lhs, used); collect_vreg(rhs, used); }
        IrInst::UnaryOp { src, .. } => { collect_vreg(src, used); }
        IrInst::Copy { src, .. } => { collect_vreg(src, used); }
        IrInst::Store { addr, value, .. } => { collect_vreg(addr, used); collect_vreg(value, used); }
        IrInst::Load { addr, .. } => { collect_vreg(addr, used); }
        IrInst::AddressOf { src, .. } => { used.insert(*src); }
        IrInst::Call { args, .. } => { for arg in args { collect_vreg(arg, used); } }
        IrInst::Syscall { number, args, .. } => { collect_vreg(number, used); for arg in args { collect_vreg(arg, used); } }
        IrInst::Compare { lhs, rhs, .. } => { collect_vreg(lhs, used); collect_vreg(rhs, used); }
        IrInst::Print { value } => { collect_vreg(value, used); }
        IrInst::ArrayLoad { base, index, .. } => { collect_vreg(base, used); collect_vreg(index, used); }
        IrInst::ArrayStore { base, index, value } => { collect_vreg(base, used); collect_vreg(index, used); collect_vreg(value, used); }
        IrInst::StoreIndexed { base, index, value, .. } => { used.insert(*base); collect_vreg(index, used); collect_vreg(value, used); }
        IrInst::LoadIndexed { base, index, .. } => { used.insert(*base); collect_vreg(index, used); }
        IrInst::HashOp { addr, len, .. } => { collect_vreg(addr, used); collect_vreg(len, used); }
        IrInst::FileRead { fd, buf, len, .. } => { collect_vreg(fd, used); collect_vreg(buf, used); collect_vreg(len, used); }
        IrInst::FileClose { fd } => { collect_vreg(fd, used); }
        IrInst::StrCmp { a, b, len, .. } => { collect_vreg(a, used); collect_vreg(b, used); collect_vreg(len, used); }
        IrInst::StrLen { addr, .. } => { collect_vreg(addr, used); }
        IrInst::StrCopy { dst, src, len } => { collect_vreg(dst, used); collect_vreg(src, used); collect_vreg(len, used); }
        IrInst::Outb { port, value } => { collect_vreg(port, used); collect_vreg(value, used); }
        IrInst::Inb { port, .. } => { collect_vreg(port, used); }
        IrInst::Lgdt { addr } | IrInst::Lidt { addr } | IrInst::Invlpg { addr } => { collect_vreg(addr, used); }
        IrInst::Wrmsr { msr, value } => { collect_vreg(msr, used); collect_vreg(value, used); }
        IrInst::Rdmsr { msr, .. } => { collect_vreg(msr, used); }
        IrInst::WriteCr { cr, value } => { collect_vreg(cr, used); collect_vreg(value, used); }
        IrInst::ReadCr { cr, .. } => { collect_vreg(cr, used); }
        IrInst::Ltr { selector } => { collect_vreg(selector, used); }
        IrInst::StoreAbs { addr, value, .. } => { collect_vreg(addr, used); collect_vreg(value, used); }
        IrInst::LoadAbs { addr, .. } => { collect_vreg(addr, used); }
        IrInst::Alloca { .. } | IrInst::PrintStr { .. } | IrInst::ArrayAlloc { .. } | IrInst::FileOpen { .. } | IrInst::ArrayLength { .. } | IrInst::Nop | IrInst::Cli | IrInst::Sti | IrInst::Iretq => {}
    }
}

fn collect_values_in_terminator(term: &Terminator, used: &mut HashSet<VReg>) {
    match term {
        Terminator::Return(Some(val)) => collect_vreg(val, used),
        Terminator::Halt(val) => collect_vreg(val, used),
        Terminator::Branch { cond, .. } => { used.insert(*cond); }
        Terminator::Return(None) | Terminator::Jump(_) | Terminator::Unreachable => {}
    }
}

fn collect_vreg(val: &IrValue, used: &mut HashSet<VReg>) {
    match val {
        IrValue::Reg(r) => { used.insert(*r); }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jstar::grammar::JStarType;
    use std::collections::HashMap;

    fn make_program(instructions: Vec<IrInst>, terminator: Terminator) -> IrProgram {
        IrProgram {
            functions: vec![IrFunction {
                name: "test".to_string(),
                return_type: JStarType::Long,
                param_vregs: Vec::new(),
                blocks: vec![BasicBlock {
                    label: "entry".to_string(),
                    instructions,
                    terminator,
                }],
                next_vreg: 100,
                param_count: 0,
            }],
            string_data: Vec::new(),
            global_data: Vec::new(),
            global_vars: HashMap::new(),
            global_vregs: HashMap::new(),
        }
    }

    #[test]
    fn test_nop_removal() {
        let mut program = make_program(
            vec![
                IrInst::Nop,
                IrInst::Print { value: IrValue::Imm(42) },
                IrInst::Nop,
            ],
            Terminator::Return(None),
        );
        optimize(&mut program);
        let insts = &program.functions[0].blocks[0].instructions;
        assert!(insts.iter().all(|i| !matches!(i, IrInst::Nop)), "Nop instructions should be removed");
        assert_eq!(insts.len(), 1, "Only the Print should remain");
    }

    #[test]
    fn test_constant_folding_add() {
        let mut program = make_program(
            vec![IrInst::BinOp { dest: 0, op: IrBinOp::Add, lhs: IrValue::Imm(3), rhs: IrValue::Imm(4), ty: JStarType::Long }],
            Terminator::Return(Some(IrValue::Reg(0))),
        );
        optimize(&mut program);
        match &program.functions[0].blocks[0].terminator {
            Terminator::Return(Some(IrValue::Imm(7))) => {}
            other => panic!("Expected Return(Imm(7)), got {:?}", other),
        }
    }

    #[test]
    fn test_constant_folding_all_ops() {
        let cases: Vec<(IrBinOp, i64, i64, i64)> = vec![
            (IrBinOp::Add, 10, 20, 30), (IrBinOp::Sub, 20, 7, 13), (IrBinOp::Mul, 3, 5, 15),
            (IrBinOp::Div, 20, 4, 5), (IrBinOp::Mod, 17, 5, 2),
            (IrBinOp::And, 0xFF, 0x0F, 0x0F), (IrBinOp::Or, 0xF0, 0x0F, 0xFF),
            (IrBinOp::Xor, 0xFF, 0xFF, 0), (IrBinOp::Shl, 1, 4, 16), (IrBinOp::Shr, 16, 4, 1),
        ];
        for (op, l, r, expected) in cases {
            let mut program = make_program(
                vec![IrInst::BinOp { dest: 0, op, lhs: IrValue::Imm(l), rhs: IrValue::Imm(r), ty: JStarType::Long }],
                Terminator::Return(Some(IrValue::Reg(0))),
            );
            optimize(&mut program);
            match &program.functions[0].blocks[0].terminator {
                Terminator::Return(Some(IrValue::Imm(val))) => {
                    assert_eq!(*val, expected, "Failed for {:?}: {} op {} = {}, got {}", op, l, r, expected, val);
                }
                other => panic!("Expected Return(Imm({})) for {:?}, got {:?}", expected, op, other),
            }
        }
    }

    #[test]
    fn test_constant_folding_div_by_zero_preserved() {
        let mut program = make_program(
            vec![IrInst::BinOp { dest: 0, op: IrBinOp::Div, lhs: IrValue::Imm(10), rhs: IrValue::Imm(0), ty: JStarType::Long }],
            Terminator::Return(Some(IrValue::Reg(0))),
        );
        optimize(&mut program);
        assert!(matches!(&program.functions[0].blocks[0].instructions[0], IrInst::BinOp { .. }), "Div by zero should NOT be folded");
    }

    #[test]
    fn test_copy_propagation() {
        let mut program = make_program(
            vec![
                IrInst::Copy { dest: 0, src: IrValue::Imm(42), ty: JStarType::Long },
                IrInst::BinOp { dest: 1, op: IrBinOp::Add, lhs: IrValue::Reg(0), rhs: IrValue::Imm(8), ty: JStarType::Long },
            ],
            Terminator::Return(Some(IrValue::Reg(1))),
        );
        optimize(&mut program);
        match &program.functions[0].blocks[0].terminator {
            Terminator::Return(Some(IrValue::Imm(50))) => {}
            other => panic!("Expected Return(Imm(50)) after prop+fold, got {:?}", other),
        }
    }

    #[test]
    fn test_dead_code_elimination_unused_dest() {
        let mut program = make_program(
            vec![
                IrInst::Copy { dest: 0, src: IrValue::Imm(99), ty: JStarType::Long },
                IrInst::Copy { dest: 1, src: IrValue::Imm(1), ty: JStarType::Long },
            ],
            Terminator::Return(Some(IrValue::Reg(1))),
        );
        optimize(&mut program);
        match &program.functions[0].blocks[0].terminator {
            Terminator::Return(Some(IrValue::Imm(1))) => {}
            other => panic!("Expected Return(Imm(1)), got {:?}", other),
        }
        let insts = &program.functions[0].blocks[0].instructions;
        assert!(insts.is_empty(), "All instructions should be eliminated after propagation");
    }

    #[test]
    fn test_side_effects_not_eliminated() {
        let mut program = make_program(
            vec![
                IrInst::Print { value: IrValue::Imm(42) },
                IrInst::Store { addr: IrValue::Imm(0), value: IrValue::Imm(1), ty: JStarType::Long },
            ],
            Terminator::Return(None),
        );
        optimize(&mut program);
        assert_eq!(program.functions[0].blocks[0].instructions.len(), 2, "Side-effecting instructions must not be eliminated");
    }
}
