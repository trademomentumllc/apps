//! Token-to-Instruction Mapping — Phase 0 of the JStar compiler.
//!
//! Maps morphlex TokenVector fields to JStar instruction variants.
//! The POS tag determines the instruction category:
//!   Verb       → Operation (add, store, jump, compare, return)
//!   Noun       → Data declaration/reference (integer, buffer, counter)
//!   Adjective  → Type modifier (unsigned, static, mutable)
//!   Adverb     → Execution modifier (immediately, conditionally)
//!   Preposition → Addressing mode (into, from, at)
//!   Determiner → Scope/lifetime (the=global, a=local, this=self)
//!   Conjunction → Control flow join (and=seq, or=branch, if=cond)
//!   Pronoun    → Register alias (it=accumulator, that=last result)
//!
//! Resolution is deterministic: same token → same instruction, always.

use crate::types::TokenVector;

// ─── Instruction Set ────────────────────────────────────────────────────────

/// The JStar instruction set — operations derived from English verbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JStarInstruction {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,

    // Comparison
    Compare,
    Equal,
    Less,
    Greater,

    // Memory
    Load,
    Store,
    Move,
    Push,
    Pop,
    Allocate,

    // Control flow
    Jump,
    JumpIf,
    Call,
    Return,

    // Bitwise
    And,
    Or,
    Xor,
    Shift,
    Not,

    // I/O
    Print,
    Open,
    Close,

    // Arrays
    Length,

    // String operations
    StrCmp,
    StrLen,
    StrCopy,

    // Hashing
    Hash,

    // Address
    AddressOf,

    // System
    Syscall,
    Halt,
    Nop,

    // Kernel / bare-metal
    Outb,
    Inb,
    Cli,
    Sti,
    Lgdt,
    Lidt,
    Wrmsr,
    Rdmsr,
    Invlpg,
    WriteCr,
    ReadCr,
    Iretq,
    Ltr,
    StoreAbs,
    LoadAbs,
}

/// What category a token falls into in the JStar language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenCategory {
    /// Verb → operation/instruction
    Operation(JStarInstruction),
    /// Noun → data declaration or reference
    Data,
    /// Adjective → type modifier (unsigned, mutable, static, etc.)
    TypeModifier(TypeMod),
    /// Adverb → execution modifier
    ExecModifier(ExecMod),
    /// Preposition → addressing mode / relation
    Addressing(AddrMode),
    /// Determiner → scope/lifetime
    Scope(ScopeKind),
    /// Conjunction → control flow join
    ControlFlow(FlowKind),
    /// Pronoun → register/reference alias
    Register(RegAlias),
    /// Number literal
    Literal,
    /// Function definition keyword
    FunctionDef,
    /// Block-closing keyword ("end") — NOT a halt instruction
    BlockEnd,
    /// Interjection/Particle → ignored or comment marker
    Ignored,
}

/// Type modifiers derived from adjectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeMod {
    Unsigned,
    Signed,
    Long,
    Short,
    Static,
    Mutable,
    Volatile,
    Const,
    Default,
}

/// Execution modifiers derived from adverbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMod {
    Immediately,
    Conditionally,
    Repeatedly,
    Default,
}

/// Addressing modes derived from prepositions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrMode {
    Into,    // destination
    From,    // source
    At,      // direct address
    Through, // indirect/pointer
    By,      // offset/stride
    Default,
}

/// Scope/lifetime derived from determiners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Global,  // "the"
    Local,   // "a" / "an"
    SelfRef, // "this"
    Default,
}

/// Control flow joins derived from conjunctions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowKind {
    Sequence,    // "and"
    Branch,      // "or"
    Conditional, // "if"
    Loop,        // "while"
    ForLoop,     // "for"
    Default,
}

/// Register aliases derived from pronouns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegAlias {
    Accumulator, // "it"
    LastResult,  // "that"
    Counter,     // "this" (when pronoun, not determiner)
    Default,
}

// POS discriminant constants (mirrors vectorizer::pos_to_i8)
const POS_NOUN: i8 = 0;
const POS_VERB: i8 = 1;
const POS_ADJECTIVE: i8 = 2;
const POS_ADVERB: i8 = 3;
const POS_PRONOUN: i8 = 4;
const POS_PREPOSITION: i8 = 5;
const POS_CONJUNCTION: i8 = 6;
const POS_DETERMINER: i8 = 7;
const POS_INTERJECTION: i8 = 8;

/// Numeric literals — not a natural language POS.
/// Used by the JStar tokenizer for number tokens that bypass morphlex.
pub(crate) const POS_LITERAL: i8 = 10;

/// String literals — extracted before morphlex processing.
pub(crate) const POS_STRING: i8 = 11;

// ─── Keyword Hash Table ─────────────────────────────────────────────────────
//
// JStar keywords are resolved by their i32 identity hash — the same BLAKE3
// hash already stored in TokenVector.id. This means:
//   - No string comparison at resolution time
//   - No exception lists in morphology
//   - "return" resolves correctly even though morphology decomposes it to
//     re- + turn, because tv.id is BLAKE3("return"), not BLAKE3("turn")
//   - One i32 == i32 comparison. The vector IS the identity.

use std::collections::HashMap;
use std::sync::LazyLock;

/// BLAKE3 hash truncated to i32 — same algorithm as vectorizer::hash_to_i32.
fn keyword_hash(word: &str) -> i32 {
    let hash = blake3::hash(word.as_bytes());
    let bytes = hash.as_bytes();
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// The keyword table: i32 identity hash → TokenCategory.
/// Built once, lives for the lifetime of the program.
static KEYWORD_TABLE: LazyLock<HashMap<i32, TokenCategory>> = LazyLock::new(|| {
    let entries: &[(&str, TokenCategory)] = &[
        // ── Operations (verbs) ──
        ("return", TokenCategory::Operation(JStarInstruction::Return)),
        ("add", TokenCategory::Operation(JStarInstruction::Add)),
        ("sum", TokenCategory::Operation(JStarInstruction::Add)),
        ("subtract", TokenCategory::Operation(JStarInstruction::Sub)),
        ("multiply", TokenCategory::Operation(JStarInstruction::Mul)),
        ("divide",   TokenCategory::Operation(JStarInstruction::Div)),
        ("store",    TokenCategory::Operation(JStarInstruction::Store)),
        ("save",     TokenCategory::Operation(JStarInstruction::Store)),
        ("load",     TokenCategory::Operation(JStarInstruction::Load)),
        ("fetch",    TokenCategory::Operation(JStarInstruction::Load)),
        ("move",     TokenCategory::Operation(JStarInstruction::Move)),
        ("copy",     TokenCategory::Operation(JStarInstruction::Move)),
        ("jump",     TokenCategory::Operation(JStarInstruction::Jump)),
        ("goto",     TokenCategory::Operation(JStarInstruction::Jump)),
        ("call",     TokenCategory::Operation(JStarInstruction::Call)),
        ("invoke",   TokenCategory::Operation(JStarInstruction::Call)),
        ("halt",     TokenCategory::Operation(JStarInstruction::Halt)),
        ("exit",     TokenCategory::Operation(JStarInstruction::Halt)),
        ("end",      TokenCategory::BlockEnd),
        ("true",     TokenCategory::Literal),
        ("false",    TokenCategory::Literal),
        ("compare",  TokenCategory::Operation(JStarInstruction::Compare)),
        ("equal",    TokenCategory::Operation(JStarInstruction::Equal)),
        ("equals",   TokenCategory::Operation(JStarInstruction::Equal)),
        ("less",     TokenCategory::Operation(JStarInstruction::Less)),
        ("greater",  TokenCategory::Operation(JStarInstruction::Greater)),
        ("print",    TokenCategory::Operation(JStarInstruction::Print)),
        ("push",     TokenCategory::Operation(JStarInstruction::Push)),
        ("pop",      TokenCategory::Operation(JStarInstruction::Pop)),
        ("negate",   TokenCategory::Operation(JStarInstruction::Neg)),
        ("syscall",  TokenCategory::Operation(JStarInstruction::Syscall)),
        ("bitand",   TokenCategory::Operation(JStarInstruction::And)),
        ("bitor",    TokenCategory::Operation(JStarInstruction::Or)),
        ("bitxor",   TokenCategory::Operation(JStarInstruction::Xor)),
        ("bitnot",   TokenCategory::Operation(JStarInstruction::Not)),
        ("shift",    TokenCategory::Operation(JStarInstruction::Shift)),
        ("allocate", TokenCategory::Operation(JStarInstruction::Allocate)),
        ("addressof", TokenCategory::Operation(JStarInstruction::AddressOf)),
        ("open",     TokenCategory::Operation(JStarInstruction::Open)),
        ("close",    TokenCategory::Operation(JStarInstruction::Close)),
        ("length",   TokenCategory::Operation(JStarInstruction::Length)),
        ("strcmp",   TokenCategory::Operation(JStarInstruction::StrCmp)),
        ("strlen",   TokenCategory::Operation(JStarInstruction::StrLen)),
        ("strcopy",  TokenCategory::Operation(JStarInstruction::StrCopy)),
        ("hash",     TokenCategory::Operation(JStarInstruction::Hash)),
        // ── Kernel / bare-metal ──
        ("outb",     TokenCategory::Operation(JStarInstruction::Outb)),
        ("inb",      TokenCategory::Operation(JStarInstruction::Inb)),
        ("cli",      TokenCategory::Operation(JStarInstruction::Cli)),
        ("sti",      TokenCategory::Operation(JStarInstruction::Sti)),
        ("lgdt",     TokenCategory::Operation(JStarInstruction::Lgdt)),
        ("lidt",     TokenCategory::Operation(JStarInstruction::Lidt)),
        ("wrmsr",    TokenCategory::Operation(JStarInstruction::Wrmsr)),
        ("rdmsr",    TokenCategory::Operation(JStarInstruction::Rdmsr)),
        ("invlpg",   TokenCategory::Operation(JStarInstruction::Invlpg)),
        ("writecr",  TokenCategory::Operation(JStarInstruction::WriteCr)),
        ("readcr",   TokenCategory::Operation(JStarInstruction::ReadCr)),
        ("iretq",    TokenCategory::Operation(JStarInstruction::Iretq)),
        ("ltr",      TokenCategory::Operation(JStarInstruction::Ltr)),
        ("storeabs", TokenCategory::Operation(JStarInstruction::StoreAbs)),
        ("loadabs",  TokenCategory::Operation(JStarInstruction::LoadAbs)),
        // ── Scope (explicit keyword — not a determiner) ──
        ("global", TokenCategory::Scope(ScopeKind::Global)),
        // ── Data (type primitives and common nouns) ──
        ("integer", TokenCategory::Data),
        ("int", TokenCategory::Data),
        ("boolean", TokenCategory::Data),
        ("bool", TokenCategory::Data),
        ("character", TokenCategory::Data),
        ("char", TokenCategory::Data),
        ("double", TokenCategory::Data),
        ("float", TokenCategory::Data),
        ("long", TokenCategory::Data),
        ("short", TokenCategory::Data),
        ("byte", TokenCategory::Data),
        ("void", TokenCategory::Data),
        ("number", TokenCategory::Data),
        ("count", TokenCategory::Data),
        ("counter", TokenCategory::Data),
        ("result", TokenCategory::Data),
        ("value", TokenCategory::Data),
        ("buffer", TokenCategory::Data),
        ("array", TokenCategory::Data),
        // ── Determiners (scope) ──
        // Morphlex may misclassify short function words — the hash table
        // catches them by i32 identity, same pattern as "return"/"integer".
        ("a", TokenCategory::Scope(ScopeKind::Local)),
        ("an", TokenCategory::Scope(ScopeKind::Local)),
        ("the", TokenCategory::Scope(ScopeKind::Global)),
        ("this", TokenCategory::Scope(ScopeKind::SelfRef)),
        // ── Prepositions (addressing) ──
        ("into", TokenCategory::Addressing(AddrMode::Into)),
        ("from", TokenCategory::Addressing(AddrMode::From)),
        ("to", TokenCategory::Addressing(AddrMode::Into)),
        ("at", TokenCategory::Addressing(AddrMode::At)),
        ("through", TokenCategory::Addressing(AddrMode::Through)),
        // ── Pronouns (register aliases) ──
        ("it", TokenCategory::Register(RegAlias::Accumulator)),
        ("that", TokenCategory::Register(RegAlias::LastResult)),
        // ── Control flow (conjunctions) ──
        // Morphlex may classify "if"/"while" as anything — the hash table
        // ensures they always resolve to ControlFlow regardless of POS.
        ("if", TokenCategory::ControlFlow(FlowKind::Conditional)),
        ("else", TokenCategory::ControlFlow(FlowKind::Branch)),
        ("while", TokenCategory::ControlFlow(FlowKind::Loop)),
        ("for", TokenCategory::ControlFlow(FlowKind::ForLoop)),
        // ── Function definition ──
        ("define", TokenCategory::FunctionDef),
        ("function", TokenCategory::FunctionDef),
        // ── Addressing (additional) ──
        ("with", TokenCategory::Addressing(AddrMode::By)),
    ];

    let mut map = HashMap::with_capacity(entries.len());
    for (word, category) in entries {
        map.insert(keyword_hash(word), *category);
    }
    map
});

/// Check whether an i32 hash corresponds to a known JStar keyword.
pub fn is_keyword(id: i32) -> bool {
    KEYWORD_TABLE.contains_key(&id)
}

// ─── Resolution ─────────────────────────────────────────────────────────────

/// Resolve a morphlex TokenVector into a JStar token category.
///
/// Two-phase resolution:
///   1. i32 keyword lookup — check tv.id (BLAKE3 hash of the original lexeme)
///      against the pre-hashed keyword table. O(1) integer comparison.
///      No strings involved. Catches everything: "return", "integer", "add".
///   2. POS-based dispatch — the general case for non-keyword tokens.
///
/// Deterministic: same vector → same category, always.
pub fn resolve(tv: &TokenVector, lemma: &str) -> TokenCategory {
    // Phase 1: i32 keyword lookup on the original lexeme hash.
    // tv.id = BLAKE3(original_lexeme.to_lowercase()) — already computed.
    // Copy out of packed struct to avoid unaligned reference (E0793).
    let id = tv.id;
    if let Some(cat) = KEYWORD_TABLE.get(&id) {
        return *cat;
    }

    // Phase 2: POS-based dispatch.
    let lemma_lower = lemma.to_lowercase();
    match tv.pos {
        POS_VERB => TokenCategory::Operation(resolve_verb(&lemma_lower)),
        POS_NOUN => TokenCategory::Data,
        POS_ADJECTIVE => TokenCategory::TypeModifier(resolve_adjective(&lemma_lower)),
        POS_ADVERB => TokenCategory::ExecModifier(resolve_adverb(&lemma_lower)),
        POS_PREPOSITION => TokenCategory::Addressing(resolve_preposition(&lemma_lower)),
        POS_DETERMINER => TokenCategory::Scope(resolve_determiner(&lemma_lower)),
        POS_CONJUNCTION => TokenCategory::ControlFlow(resolve_conjunction(&lemma_lower)),
        POS_PRONOUN => TokenCategory::Register(resolve_pronoun(&lemma_lower)),
        POS_INTERJECTION => TokenCategory::Ignored,
        POS_LITERAL => TokenCategory::Literal,
        POS_STRING => TokenCategory::Literal, // string literals route through Literal
        _ => TokenCategory::Ignored,          // Particle, unknown
    }
}

/// Map a verb lemma to a specific JStar instruction.
fn resolve_verb(lemma: &str) -> JStarInstruction {
    match lemma {
        // Arithmetic
        "add" | "sum" | "plus" | "increase" => JStarInstruction::Add,
        "subtract" | "sub" | "minus" | "decrease" | "reduce" => JStarInstruction::Sub,
        "multiply" | "mul" | "times" => JStarInstruction::Mul,
        "divide" | "div" | "split" => JStarInstruction::Div,
        "modulo" | "mod" | "remainder" => JStarInstruction::Mod,
        "negate" | "invert" => JStarInstruction::Neg,

        // Comparison
        "compare" | "check" | "test" => JStarInstruction::Compare,
        "equal" | "equals" | "match" => JStarInstruction::Equal,

        // Memory
        "load" | "read" | "fetch" | "get" => JStarInstruction::Load,
        "store" | "write" | "save" | "put" | "set" => JStarInstruction::Store,
        "move" | "copy" | "transfer" => JStarInstruction::Move,
        "push" => JStarInstruction::Push,
        "pop" | "pull" => JStarInstruction::Pop,

        // Control flow
        "jump" | "goto" | "branch" => JStarInstruction::Jump,
        "call" | "invoke" | "execute" | "run" => JStarInstruction::Call,
        "return" | "yield" | "give" | "produce" => JStarInstruction::Return,

        // Bitwise
        "shift" | "rotate" => JStarInstruction::Shift,

        // I/O
        "print" | "show" | "display" | "output" => JStarInstruction::Print,
        "open" => JStarInstruction::Open,
        "close" | "shut" => JStarInstruction::Close,

        // Arrays
        "length" | "size" | "count" => JStarInstruction::Length,

        // Hashing
        "hash" | "digest" => JStarInstruction::Hash,

        // String operations
        "strcmp" => JStarInstruction::StrCmp,
        "strlen" => JStarInstruction::StrLen,
        "strcopy" => JStarInstruction::StrCopy,

        // System
        "halt" | "stop" | "exit" | "quit" | "end" | "terminate" => JStarInstruction::Halt,
        "syscall" | "interrupt" | "signal" => JStarInstruction::Syscall,
        "allocate" | "alloc" | "reserve" => JStarInstruction::Allocate,

        // Default: treat unknown verbs as no-op
        _ => JStarInstruction::Nop,
    }
}

/// Map an adjective lemma to a type modifier.
fn resolve_adjective(lemma: &str) -> TypeMod {
    match lemma {
        "unsigned" => TypeMod::Unsigned,
        "signed" => TypeMod::Signed,
        "long" => TypeMod::Long,
        "short" => TypeMod::Short,
        "static" => TypeMod::Static,
        "mutable" | "mut" => TypeMod::Mutable,
        "volatile" => TypeMod::Volatile,
        "constant" | "const" | "final" => TypeMod::Const,
        _ => TypeMod::Default,
    }
}

/// Map an adverb lemma to an execution modifier.
fn resolve_adverb(lemma: &str) -> ExecMod {
    match lemma {
        "immediately" | "now" | "directly" => ExecMod::Immediately,
        "conditionally" | "maybe" | "possibly" => ExecMod::Conditionally,
        "repeatedly" | "again" | "always" => ExecMod::Repeatedly,
        _ => ExecMod::Default,
    }
}

/// Map a preposition lemma to an addressing mode.
fn resolve_preposition(lemma: &str) -> AddrMode {
    match lemma {
        "into" | "to" | "onto" => AddrMode::Into,
        "from" | "out" | "off" => AddrMode::From,
        "at" | "in" | "on" => AddrMode::At,
        "through" | "via" | "across" => AddrMode::Through,
        "by" | "with" | "per" => AddrMode::By,
        _ => AddrMode::Default,
    }
}

/// Map a determiner lemma to a scope/lifetime.
fn resolve_determiner(lemma: &str) -> ScopeKind {
    match lemma {
        "the" => ScopeKind::Global,
        "a" | "an" | "some" => ScopeKind::Local,
        "this" | "these" => ScopeKind::SelfRef,
        _ => ScopeKind::Default,
    }
}

/// Map a conjunction lemma to a control flow kind.
fn resolve_conjunction(lemma: &str) -> FlowKind {
    match lemma {
        "and" | "then" | "also" => FlowKind::Sequence,
        "or" | "else" | "otherwise" => FlowKind::Branch,
        "if" | "when" | "unless" => FlowKind::Conditional,
        "while" | "until" | "during" => FlowKind::Loop,
        _ => FlowKind::Default,
    }
}

/// Map a pronoun lemma to a register alias.
fn resolve_pronoun(lemma: &str) -> RegAlias {
    match lemma {
        "it" | "itself" => RegAlias::Accumulator,
        "that" | "which" | "what" => RegAlias::LastResult,
        "this" => RegAlias::Counter,
        _ => RegAlias::Default,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TokenVector;

    /// Helper: create a TokenVector with the i32 hash of a word and a given POS.
    /// This mirrors what the vectorizer does: id = BLAKE3(word.to_lowercase()).
    fn make_tv_for(word: &str, pos: i8) -> TokenVector {
        TokenVector {
            id: keyword_hash(word),
            lemma_id: 0,
            pos,
            role: 0,
            morph: 0,
        }
    }

    #[test]
    fn test_verb_resolves_to_operation() {
        let tv = make_tv_for("add", POS_VERB);
        match resolve(&tv, "add") {
            TokenCategory::Operation(JStarInstruction::Add) => {}
            other => panic!("Expected Operation(Add), got {:?}", other),
        }
    }

    #[test]
    fn test_store_instruction() {
        let tv = make_tv_for("store", POS_VERB);
        match resolve(&tv, "store") {
            TokenCategory::Operation(JStarInstruction::Store) => {}
            other => panic!("Expected Operation(Store), got {:?}", other),
        }
    }

    #[test]
    fn test_jump_instruction() {
        let tv = make_tv_for("jump", POS_VERB);
        match resolve(&tv, "jump") {
            TokenCategory::Operation(JStarInstruction::Jump) => {}
            other => panic!("Expected Operation(Jump), got {:?}", other),
        }
    }

    #[test]
    fn test_return_via_id_hash() {
        // "return" has a real prefix (re- + turn). Morphology decomposes it,
        // so the lemma is "turn" and POS might be anything. But tv.id is
        // BLAKE3("return") — the keyword table catches it by integer lookup.
        let tv = make_tv_for("return", POS_NOUN);
        match resolve(&tv, "turn") {
            TokenCategory::Operation(JStarInstruction::Return) => {}
            other => panic!("Expected Operation(Return), got {:?}", other),
        }
    }

    #[test]
    fn test_integer_via_id_hash() {
        // "integer" gets decomposed to in- + teg + -er by morphology.
        // POS ends up as Adjective, lemma is "teg". But tv.id is
        // BLAKE3("integer") — resolved to Data by one i32 comparison.
        let tv = make_tv_for("integer", POS_ADJECTIVE);
        match resolve(&tv, "teg") {
            TokenCategory::Data => {}
            other => panic!("Expected Data, got {:?}", other),
        }
    }

    #[test]
    fn test_boolean_via_id_hash() {
        let tv = make_tv_for("boolean", POS_ADJECTIVE);
        match resolve(&tv, "boolean") {
            TokenCategory::Data => {}
            other => panic!("Expected Data, got {:?}", other),
        }
    }

    #[test]
    fn test_noun_resolves_to_data() {
        // "dog" is not a keyword — falls through to POS-based dispatch.
        let tv = make_tv_for("dog", POS_NOUN);
        match resolve(&tv, "dog") {
            TokenCategory::Data => {}
            other => panic!("Expected Data, got {:?}", other),
        }
    }

    #[test]
    fn test_adjective_resolves_to_type_modifier() {
        // "unsigned" is not in the keyword table — falls to POS dispatch.
        let tv = make_tv_for("unsigned", POS_ADJECTIVE);
        match resolve(&tv, "unsigned") {
            TokenCategory::TypeModifier(TypeMod::Unsigned) => {}
            other => panic!("Expected TypeModifier(Unsigned), got {:?}", other),
        }
    }

    #[test]
    fn test_determiner_scope() {
        let tv = make_tv_for("the", POS_DETERMINER);
        match resolve(&tv, "the") {
            TokenCategory::Scope(ScopeKind::Global) => {}
            other => panic!("Expected Scope(Global), got {:?}", other),
        }
        let tv = make_tv_for("a", POS_DETERMINER);
        match resolve(&tv, "a") {
            TokenCategory::Scope(ScopeKind::Local) => {}
            other => panic!("Expected Scope(Local), got {:?}", other),
        }
    }

    #[test]
    fn test_conjunction_control_flow() {
        let tv = make_tv_for("if", POS_CONJUNCTION);
        match resolve(&tv, "if") {
            TokenCategory::ControlFlow(FlowKind::Conditional) => {}
            other => panic!("Expected ControlFlow(Conditional), got {:?}", other),
        }
    }

    #[test]
    fn test_preposition_addressing() {
        let tv = make_tv_for("into", POS_PREPOSITION);
        match resolve(&tv, "into") {
            TokenCategory::Addressing(AddrMode::Into) => {}
            other => panic!("Expected Addressing(Into), got {:?}", other),
        }
    }

    #[test]
    fn test_pronoun_register() {
        let tv = make_tv_for("it", POS_PRONOUN);
        match resolve(&tv, "it") {
            TokenCategory::Register(RegAlias::Accumulator) => {}
            other => panic!("Expected Register(Accumulator), got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_verb_is_nop() {
        let tv = make_tv_for("flibbertigibbet", POS_VERB);
        match resolve(&tv, "flibbertigibbet") {
            TokenCategory::Operation(JStarInstruction::Nop) => {}
            other => panic!("Expected Operation(Nop), got {:?}", other),
        }
    }

    #[test]
    fn test_determinism() {
        let tv = make_tv_for("add", POS_VERB);
        let a = resolve(&tv, "add");
        let b = resolve(&tv, "add");
        assert_eq!(a, b);
    }
}
