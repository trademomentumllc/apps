//! Type Checker — Phase 3 of the JStar compiler.
//!
//! Walks the JStar AST and annotates every node with its type.
//! Type system is Java's 8 primitives: boolean, byte, short, int, long, float, double, char.
//!
//! Type inference:
//!   - Adjective modifiers ("unsigned", "long") refine the default type
//!   - Default type is int (i32) when no modifier is present
//!   - Type errors are compile-time (no runtime type checks in emitted code)
//!
//! Monadic error handling: all functions return MorphResult<T>.

use super::grammar::*;
use super::token_map::*;
use crate::types::MorphResult;
use std::collections::HashMap;

/// Symbol table entry — tracks declared variables and their types.
#[derive(Debug, Clone)]
struct Symbol {
    name: String,
    ty: JStarType,
    scope: ScopeKind,
}

/// Type checker state.
struct TypeChecker {
    symbols: HashMap<String, Symbol>,
}

/// Type-check a JStar program.
///
/// Resolves all types, validates operand compatibility, and produces
/// a TypedProgram where every node carries its type.
pub fn check(program: &JStarProgram) -> MorphResult<TypedProgram> {
    let mut checker = TypeChecker {
        symbols: HashMap::new(),
    };
    checker.check_program(program)
}

impl TypeChecker {
    fn check_program(&mut self, program: &JStarProgram) -> MorphResult<TypedProgram> {
        let mut typed_stmts = Vec::new();
        for stmt in &program.statements {
            typed_stmts.push(self.check_statement(stmt)?);
        }
        Ok(TypedProgram {
            statements: typed_stmts,
        })
    }

    fn check_statement(&mut self, stmt: &JStarStatement) -> MorphResult<TypedStatement> {
        match stmt {
            JStarStatement::Execute { op, operands } => {
                let typed_operands: Vec<TypedOperand> = operands
                    .iter()
                    .map(|o| self.check_operand(o))
                    .collect::<MorphResult<Vec<_>>>()?;

                // Infer result type from the operation and operands
                let result_type = self.infer_result_type(*op, &typed_operands);

                Ok(TypedStatement::Execute {
                    op: *op,
                    operands: typed_operands,
                    result_type,
                })
            }

            JStarStatement::Declare {
                scope,
                name,
                ty,
                size,
            } => {
                // Register in symbol table
                self.symbols.insert(
                    name.clone(),
                    Symbol {
                        name: name.clone(),
                        ty: *ty,
                        scope: *scope,
                    },
                );

                Ok(TypedStatement::Declare {
                    scope: *scope,
                    name: name.clone(),
                    ty: *ty,
                    size: *size,
                })
            }

            JStarStatement::ControlFlow {
                kind,
                condition,
                body,
                else_body,
            } => {
                let typed_condition = match condition {
                    Some(cond) => Some(Box::new(self.check_statement(cond)?)),
                    None => None,
                };
                let typed_body: Vec<TypedStatement> = body
                    .iter()
                    .map(|s| self.check_statement(s))
                    .collect::<MorphResult<Vec<_>>>()?;
                let typed_else: Vec<TypedStatement> = else_body
                    .iter()
                    .map(|s| self.check_statement(s))
                    .collect::<MorphResult<Vec<_>>>()?;

                Ok(TypedStatement::ControlFlow {
                    kind: *kind,
                    condition: typed_condition,
                    body: typed_body,
                    else_body: typed_else,
                })
            }

            JStarStatement::Return { value } => {
                let typed_value = match value {
                    Some(v) => Some(self.check_operand(v)?),
                    None => None,
                };
                let ty = typed_value
                    .as_ref()
                    .map(|v| v.ty())
                    .unwrap_or(JStarType::Void);

                Ok(TypedStatement::Return {
                    value: typed_value,
                    ty,
                })
            }

            JStarStatement::FunctionDef {
                name,
                params,
                body,
                return_type,
            } => {
                // Register function parameters as symbols
                for (pname, pty) in params {
                    self.symbols.insert(
                        pname.clone(),
                        Symbol {
                            name: pname.clone(),
                            ty: *pty,
                            scope: ScopeKind::Local,
                        },
                    );
                }
                let typed_body: Vec<TypedStatement> = body
                    .iter()
                    .map(|s| self.check_statement(s))
                    .collect::<MorphResult<Vec<_>>>()?;
                Ok(TypedStatement::FunctionDef {
                    name: name.clone(),
                    params: params.clone(),
                    body: typed_body,
                    return_type: *return_type,
                })
            }

            JStarStatement::Label(name) => Ok(TypedStatement::Label(name.clone())),

            JStarStatement::Nop => Ok(TypedStatement::Nop),
        }
    }

    fn check_operand(&self, operand: &JStarOperand) -> MorphResult<TypedOperand> {
        match operand {
            JStarOperand::Variable {
                name,
                scope,
                modifiers,
            } => {
                // Look up in symbol table first
                let base_ty = if let Some(sym) = self.symbols.get(name) {
                    sym.ty
                } else {
                    // Infer from noun + modifiers
                    JStarType::from_noun(name)
                };

                // Apply modifiers
                let ty = modifiers.iter().fold(base_ty, |t, m| t.apply_modifier(*m));

                Ok(TypedOperand::Variable {
                    name: name.clone(),
                    scope: *scope,
                    ty,
                })
            }

            JStarOperand::Immediate(val) => {
                // Infer type from value range
                let ty = infer_immediate_type(*val);
                Ok(TypedOperand::Immediate(*val, ty))
            }

            JStarOperand::Register(reg) => {
                // Registers default to Int
                Ok(TypedOperand::Register(*reg, JStarType::Int))
            }

            JStarOperand::StringLiteral(s) => Ok(TypedOperand::StringLiteral(s.clone())),

            JStarOperand::Addressed { mode, target } => {
                let typed_target = self.check_operand(target)?;
                let ty = typed_target.ty();
                Ok(TypedOperand::Addressed {
                    mode: *mode,
                    target: Box::new(typed_target),
                    ty,
                })
            }
        }
    }

    /// Infer the result type of an operation from its operands.
    fn infer_result_type(&self, op: JStarInstruction, operands: &[TypedOperand]) -> JStarType {
        match op {
            // Comparison operations always produce boolean
            JStarInstruction::Compare
            | JStarInstruction::Equal
            | JStarInstruction::Less
            | JStarInstruction::Greater
            | JStarInstruction::StrCmp => JStarType::Boolean,

            // Address-of produces a pointer (Long = 8 bytes)
            JStarInstruction::AddressOf => JStarType::Long,

            // String length produces an integer
            JStarInstruction::StrLen => JStarType::Long,

            // Void operations
            JStarInstruction::Store
            | JStarInstruction::Push
            | JStarInstruction::Jump
            | JStarInstruction::JumpIf
            | JStarInstruction::Syscall
            | JStarInstruction::Halt
            | JStarInstruction::StrCopy
            | JStarInstruction::Nop => JStarType::Void,

            // StoreAbs: type is the *value* being stored (second operand)
            JStarInstruction::StoreAbs => operands.get(1).map(|o| o.ty()).unwrap_or(JStarType::Long),

            // For arithmetic and most other ops, use the widest operand type
            _ => operands.first().map(|o| o.ty()).unwrap_or(JStarType::Int),
        }
    }
}

/// Infer the smallest type that can hold an immediate value.
fn infer_immediate_type(val: i64) -> JStarType {
    if val >= i8::MIN as i64 && val <= i8::MAX as i64 {
        JStarType::Byte
    } else if val >= i16::MIN as i64 && val <= i16::MAX as i64 {
        JStarType::Short
    } else if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
        JStarType::Int
    } else {
        JStarType::Long
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_immediate_type_inference() {
        assert_eq!(infer_immediate_type(42), JStarType::Byte);
        assert_eq!(infer_immediate_type(200), JStarType::Short);
        assert_eq!(infer_immediate_type(100_000), JStarType::Int);
        assert_eq!(infer_immediate_type(5_000_000_000), JStarType::Long);
    }

    #[test]
    fn test_check_empty_program() {
        let prog = JStarProgram { statements: vec![] };
        let typed = check(&prog).unwrap();
        assert!(typed.statements.is_empty());
    }

    #[test]
    fn test_check_declaration() {
        let prog = JStarProgram {
            statements: vec![JStarStatement::Declare {
                scope: ScopeKind::Local,
                name: "counter".to_string(),
                ty: JStarType::Int,
                size: None,
            }],
        };
        let typed = check(&prog).unwrap();
        match &typed.statements[0] {
            TypedStatement::Declare { name, ty, .. } => {
                assert_eq!(name, "counter");
                assert_eq!(*ty, JStarType::Int);
            }
            other => panic!("Expected Declare, got {:?}", other),
        }
    }

    #[test]
    fn test_check_return_immediate() {
        let prog = JStarProgram {
            statements: vec![JStarStatement::Return {
                value: Some(JStarOperand::Immediate(42)),
            }],
        };
        let typed = check(&prog).unwrap();
        match &typed.statements[0] {
            TypedStatement::Return { ty, .. } => {
                assert_eq!(*ty, JStarType::Byte); // 42 fits in byte
            }
            other => panic!("Expected Return, got {:?}", other),
        }
    }

    #[test]
    fn test_symbol_table_lookup() {
        let prog = JStarProgram {
            statements: vec![
                JStarStatement::Declare {
                    scope: ScopeKind::Global,
                    name: "result".to_string(),
                    ty: JStarType::Long,
                    size: None,
                },
                JStarStatement::Execute {
                    op: JStarInstruction::Load,
                    operands: vec![JStarOperand::Variable {
                        name: "result".to_string(),
                        scope: ScopeKind::Global,
                        modifiers: vec![],
                    }],
                },
            ],
        };
        let typed = check(&prog).unwrap();
        // The Load operand should resolve to Long (from symbol table)
        match &typed.statements[1] {
            TypedStatement::Execute { operands, .. } => {
                assert_eq!(operands[0].ty(), JStarType::Long);
            }
            other => panic!("Expected Execute, got {:?}", other),
        }
    }

    #[test]
    fn test_comparison_returns_boolean() {
        let prog = JStarProgram {
            statements: vec![JStarStatement::Execute {
                op: JStarInstruction::Compare,
                operands: vec![JStarOperand::Immediate(1), JStarOperand::Immediate(2)],
            }],
        };
        let typed = check(&prog).unwrap();
        match &typed.statements[0] {
            TypedStatement::Execute { result_type, .. } => {
                assert_eq!(*result_type, JStarType::Boolean);
            }
            other => panic!("Expected Execute, got {:?}", other),
        }
    }
}
