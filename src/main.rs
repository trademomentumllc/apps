//! morphlex CLI -- compile English into a deterministic vector database.
//! PQC encrypted: ML-KEM-1024 (FIPS 203) + ML-DSA-87 (FIPS 204) + SLH-DSA (FIPS 205) + AES-256-GCM.
//! Key derivation: HKDF-SHA3-512.

use clap::{Parser, Subcommand};
use morphlex::vectorizer;
use std::fs::OpenOptions;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "morphlex")]
#[command(about = "Deterministic natural language tokenizer and vector compiler")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a word list file into a PQC-encrypted vector database
    Compile {
        /// Path to input word list (one word per line)
        #[arg(short, long)]
        input: PathBuf,

        /// Output path for the encrypted database
        #[arg(short, long, default_value = "morphlex.db.enc")]
        output: PathBuf,
    },

    /// Compile the system dictionary (/usr/share/dict/words)
    CompileDict {
        /// Output path for the encrypted database
        #[arg(short, long, default_value = "morphlex.db.enc")]
        output: PathBuf,
    },

    /// Tokenize and vectorize a text string (for testing/inspection)
    Tokenize {
        /// The text to tokenize
        text: String,
    },

    /// Inspect a PQC-encrypted database (requires key bundle directory)
    Inspect {
        /// Path to the encrypted database
        #[arg(short, long)]
        database: PathBuf,

        /// Path to the key bundle directory (contains dk.bin, vk.bin)
        #[arg(short, long)]
        keys: PathBuf,
    },

    /// Build a search index from text files
    Index {
        /// Input files to index
        #[arg(short, long, num_args = 1..)]
        input: Vec<PathBuf>,

        /// Output path for the search index
        #[arg(short, long, default_value = "morphlex.mxidx")]
        output: PathBuf,

        /// Store original text in the index (for snippet extraction)
        #[arg(long, default_value_t = false)]
        store_text: bool,
    },

    /// Search an index for matching documents
    Search {
        /// Path to the search index
        #[arg(short = 'x', long, default_value = "morphlex.mxidx")]
        index: PathBuf,

        /// The search query
        #[arg(short, long)]
        query: String,

        /// Query mode: "all" (intersection) or "any" (union)
        #[arg(short, long, default_value = "all")]
        mode: String,

        /// Filter by POS tag (0-9)
        #[arg(long)]
        pos: Option<i8>,

        /// Filter by semantic role (0-8)
        #[arg(long)]
        role: Option<i8>,

        /// Maximum number of results
        #[arg(long, default_value_t = 20)]
        max_results: usize,
    },

    /// Compile a JStar source file (.jstr) to a native ELF binary
    Jstar {
        #[command(subcommand)]
        action: JStarAction,
    },

    /// Launch the JStar shell (interactive REPL or script execution)
    Jsh {
        /// Optional .jsh script file to execute (omit for REPL mode)
        script: Option<PathBuf>,
    },

    /// Crawl a website recursively, extract text as markdown
    Crawl {
        /// Starting URL to crawl
        url: String,

        /// Maximum link depth from the seed URL
        #[arg(short, long, default_value = "3")]
        depth: u32,

        /// Delay between requests in milliseconds
        #[arg(long, default_value = "1000")]
        delay: u64,

        /// Output directory for markdown files
        #[arg(short, long, default_value = "./crawl_data/")]
        output: PathBuf,

        /// Maximum number of pages to crawl (0 = unlimited)
        #[arg(long, default_value = "0")]
        max_pages: u32,

        /// User-agent string for HTTP requests
        #[arg(long, default_value = "morphlex-crawler/0.1")]
        user_agent: String,
    },

    /// Rational Reserve (swaRRm) multi-agent system commands
    Rr {
        #[command(subcommand)]
        action: RrAction,
    },

    /// System daemon management commands
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },

    /// LLM training and inference commands
    Llm {
        #[command(subcommand)]
        action: LlmAction,
    },
}

#[derive(Subcommand)]
enum JStarAction {
    /// Compile a .jstr source file to a native binary
    Compile {
        /// Path to the .jstr source file
        #[arg(short, long)]
        input: PathBuf,

        /// Additional source files to include (concatenated before main input)
        #[arg(long)]
        include: Vec<PathBuf>,

        /// Output path for the binary
        #[arg(short, long, default_value = "a.out")]
        output: PathBuf,

        /// Use raw tokenization (bypass NLP pipeline, for self-hosting verification)
        #[arg(long)]
        raw: bool,
    },

    /// Compile a .jstr source file to a Multiboot2 kernel ELF (bare-metal)
    Kernel {
        /// Path to the .jstr kernel source file
        #[arg(short, long)]
        input: PathBuf,

        /// Output path for the kernel ELF
        #[arg(short, long, default_value = "kernel.elf")]
        output: PathBuf,
    },

    /// Parse a .jstr source file and display the AST (for debugging)
    Parse {
        /// Path to the .jstr source file, or inline text with --text
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Inline JStar text to parse
        #[arg(short, long)]
        text: Option<String>,
    },
}

#[derive(Subcommand)]
enum RrAction {
    /// Deploy a new swarm for a mission
    Deploy {
        /// Mission objective (natural language)
        #[arg(short, long)]
        mission: String,

        /// Mission priority (routine, priority, urgent)
        #[arg(long, default_value = "routine")]
        priority: String,

        /// Database path for persistence
        #[arg(short, long, default_value = "rr_db.json")]
        db: PathBuf,
    },

    /// Get status of a swarm
    Status {
        /// Swarm ID
        #[arg(short, long)]
        swarm_id: String,

        /// Database path
        #[arg(short, long, default_value = "rr_db.json")]
        db: PathBuf,
    },

    /// List all active swarms
    List {
        /// Database path
        #[arg(short, long, default_value = "rr_db.json")]
        db: PathBuf,
    },

    /// Issue FRAGO (mission adjustment) to a swarm
    Frago {
        /// Swarm ID
        #[arg(short, long)]
        swarm_id: String,

        /// FRAGO content (new instructions)
        #[arg(short, long)]
        content: String,

        /// Database path
        #[arg(short, long, default_value = "rr_db.json")]
        db: PathBuf,
    },

    /// Disband a swarm and generate AAR
    Disband {
        /// Swarm ID
        #[arg(short, long)]
        swarm_id: String,

        /// Database path
        #[arg(short, long, default_value = "rr_db.json")]
        db: PathBuf,
    },

    /// Create a new agent
    CreateAgent {
        /// Agent type (simple, multimodal, langchain, guardian, coding, analysis, search, datamgmt, filtration)
        #[arg(short, long)]
        agent_type: String,

        /// Agent name
        #[arg(short, long)]
        name: Option<String>,

        /// Database path
        #[arg(short, long, default_value = "rr_db.json")]
        db: PathBuf,
    },

    /// Show database statistics
    Stats {
        /// Database path
        #[arg(short, long, default_value = "rr_db.json")]
        db: PathBuf,
    },
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Start System Integrity Daemon
    Integrity {
        /// Paths to monitor
        #[arg(long, num_args = 1..)]
        monitor: Vec<PathBuf>,

        /// Check interval in seconds
        #[arg(long, default_value = "60")]
        check_interval: u64,
    },

    /// Start Threat Intelligence Manager
    Threat {
        /// Scan interval in seconds
        #[arg(long, default_value = "30")]
        scan_interval: u64,
    },

    /// Run Morphogenetic Maintenance
    Maintenance {
        /// Run maintenance now (ignore schedule)
        #[arg(long)]
        run: bool,

        /// Show maintenance report only
        #[arg(long)]
        report: bool,
    },

    /// Start all daemons
    StartAll {
        /// Run in foreground (don't daemonize)
        #[arg(long)]
        foreground: bool,
    },
}

#[derive(Subcommand)]
enum LlmAction {
    /// Train a new MorphlexLLM model
    Train {
        /// Training data file (text or .db)
        #[arg(short, long)]
        data: PathBuf,

        /// Model size (small, medium, large)
        #[arg(long, default_value = "small")]
        size: String,

        /// Number of epochs
        #[arg(long, default_value = "10")]
        epochs: usize,

        /// Batch size
        #[arg(long, default_value = "32")]
        batch_size: usize,

        /// Learning rate
        #[arg(long, default_value = "0.0001")]
        lr: f32,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "llm_checkpoints")]
        output: String,
    },

    /// Export trained model to GGUF format
    Export {
        /// Model checkpoint file
        #[arg(short, long)]
        model: PathBuf,

        /// Output GGUF file
        #[arg(short, long)]
        output: PathBuf,

        /// Quantize to F16
        #[arg(long)]
        quantize: bool,
    },

    /// Run inference with trained model
    Infer {
        /// Model checkpoint file
        #[arg(short, long)]
        model: PathBuf,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Max tokens to generate
        #[arg(long, default_value = "100")]
        max_tokens: usize,

        /// Temperature for sampling
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },

    /// Show model information
    Info {
        /// Model checkpoint file
        #[arg(short, long)]
        model: PathBuf,
    },
}

/// Write a PQC key bundle to a directory.
///
/// Writes all keys needed for decryption and signature verification:
///   dk.bin      -- ML-KEM-1024 decapsulation key seed (64 bytes)
///   sk.bin      -- ML-DSA-87 signing key seed (32 bytes)
///   vk.bin      -- ML-DSA-87 verifying key (2,592 bytes)
///   slh_sk.bin  -- SLH-DSA-SHAKE-256s signing key (128 bytes)
///   slh_vk.bin  -- SLH-DSA-SHAKE-256s verifying key (64 bytes)
fn write_key_file(path: &Path, bytes: &[u8], mode: u32, label: &str) {
    #[cfg(unix)]
    {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(mode)
            .open(path)
            .unwrap_or_else(|_| panic!("Failed to write {label}"));
        file.write_all(bytes)
            .unwrap_or_else(|_| panic!("Failed to write {label}"));
        file.sync_all()
            .unwrap_or_else(|_| panic!("Failed to flush {label}"));
    }

    #[cfg(not(unix))]
    {
        let _ = mode;
        std::fs::write(path, bytes).unwrap_or_else(|_| panic!("Failed to write {label}"));
    }
}

fn write_key_bundle(bundle: &morphlex::database::PqcKeyBundle, dir: &std::path::Path) {
    std::fs::create_dir_all(dir).expect("Failed to create key directory");

    #[cfg(unix)]
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
        .expect("Failed to harden key directory permissions");

    write_key_file(
        &dir.join("dk.bin"),
        &bundle.decapsulation_key,
        0o600,
        "decapsulation key",
    );
    write_key_file(
        &dir.join("sk.bin"),
        &bundle.signing_key,
        0o600,
        "ML-DSA-87 signing key",
    );
    write_key_file(
        &dir.join("vk.bin"),
        &bundle.verifying_key,
        0o644,
        "ML-DSA-87 verifying key",
    );
    write_key_file(
        &dir.join("slh_sk.bin"),
        &bundle.slh_signing_key,
        0o600,
        "SLH-DSA signing key",
    );
    write_key_file(
        &dir.join("slh_vk.bin"),
        &bundle.slh_verifying_key,
        0o644,
        "SLH-DSA verifying key",
    );
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { input, output } => {
            let content = std::fs::read_to_string(&input).expect("Failed to read input file");
            let words: Vec<String> = content.lines().map(|l| l.trim().to_string()).collect();

            let db_path = output.with_extension("db");
            let bundle =
                morphlex::compile_lexicon(&words, &db_path, &output).expect("Compilation failed");

            let key_dir = output.with_extension("keys");
            write_key_bundle(&bundle, &key_dir);

            println!("Compiled {} words -> {}", words.len(), output.display());
            println!("PQC keys: {}", key_dir.display());
            println!("Signature: {}", output.with_extension("sig").display());

            let _ = std::fs::remove_file(&db_path);
        }

        Commands::CompileDict { output } => {
            let dict_path = "/usr/share/dict/words";
            let content = std::fs::read_to_string(dict_path)
                .expect("System dictionary not found at /usr/share/dict/words");
            let words: Vec<String> = content.lines().map(|l| l.trim().to_string()).collect();

            println!("Compiling {} words...", words.len());

            let db_path = output.with_extension("db");
            let bundle =
                morphlex::compile_lexicon(&words, &db_path, &output).expect("Compilation failed");

            let key_dir = output.with_extension("keys");
            write_key_bundle(&bundle, &key_dir);

            println!("Compiled -> {}", output.display());
            println!("PQC keys: {}", key_dir.display());
            println!("Signature: {}", output.with_extension("sig").display());

            let _ = std::fs::remove_file(&db_path);
        }

        Commands::Tokenize { text } => {
            let (lemmas, vectors) = morphlex::compile(&text).expect("Tokenization failed");

            for (lemma, tv) in lemmas.iter().zip(vectors.iter()) {
                let id = tv.id;
                let lemma_id = tv.lemma_id;
                let pos = tv.pos;
                let role = tv.role;
                let morph = tv.morph;
                println!("-----------------------------");
                println!("  lemma:    {}", lemma);
                println!("  id:       {} (0x{:08X})", id, id as u32);
                println!("  lemma_id: {} (0x{:08X})", lemma_id, lemma_id as u32);
                println!("  pos:      {:?}", vectorizer::i8_to_pos(pos));
                println!("  role:     {:?}", vectorizer::i8_to_role(role));
                println!("  morph:    0b{:016b}", morph);
                println!("  bytes:    {:?}", tv.to_bytes());
            }
            println!("-----------------------------");
            println!(
                "{} tokens, {} bytes total",
                vectors.len(),
                vectors.len() * 12
            );
        }

        Commands::Index {
            input,
            output,
            store_text,
        } => {
            let mut index = morphlex::search::SearchIndex::new();
            let mut file_count = 0;

            for path in &input {
                if path.is_dir() {
                    // Index all .txt files in directory
                    if let Ok(entries) = std::fs::read_dir(path) {
                        for entry in entries.flatten() {
                            let p = entry.path();
                            if p.extension().map_or(false, |e| e == "txt") {
                                let content =
                                    std::fs::read_to_string(&p).expect("Failed to read file");
                                let title = p
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                if store_text {
                                    index
                                        .add_document_with_text(&title, &content)
                                        .expect("Failed to index document");
                                } else {
                                    index
                                        .add_document(&title, &content)
                                        .expect("Failed to index document");
                                }
                                file_count += 1;
                            }
                        }
                    }
                } else {
                    let content = std::fs::read_to_string(path).expect("Failed to read file");
                    let title = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if store_text {
                        index
                            .add_document_with_text(&title, &content)
                            .expect("Failed to index document");
                    } else {
                        index
                            .add_document(&title, &content)
                            .expect("Failed to index document");
                    }
                    file_count += 1;
                }
            }

            let size = index.write_to_path(&output).expect("Failed to write index");
            println!(
                "Indexed {} files, {} postings -> {} ({} bytes)",
                file_count,
                index.posting_count(),
                output.display(),
                size
            );
        }

        Commands::Search {
            index: index_path,
            query,
            mode,
            pos,
            role,
            max_results,
        } => {
            let index = morphlex::search::SearchIndex::read_from_path(&index_path)
                .expect("Failed to read index");

            let query_mode = match mode.as_str() {
                "any" => morphlex::types::QueryMode::Any,
                _ => morphlex::types::QueryMode::All,
            };

            let config = morphlex::types::SearchConfig {
                mode: query_mode,
                filter: morphlex::types::SearchFilter {
                    pos,
                    role,
                    morph_mask: None,
                },
                max_results,
            };

            let results = morphlex::search::search(&index, &query, &config).expect("Search failed");

            if results.is_empty() {
                println!("No results found.");
            } else {
                for (i, r) in results.iter().enumerate() {
                    let title = index
                        .get_doc(r.doc_id)
                        .map(|m| m.title.as_str())
                        .unwrap_or("unknown");
                    println!(
                        "[{:>3}] score={:<6} doc_id={:<12} title={}",
                        i + 1,
                        r.score,
                        r.doc_id,
                        title
                    );
                    if let Some(text) = index.get_doc_text(r.doc_id) {
                        let snippet: String = text.chars().take(80).collect();
                        println!("      {}", snippet);
                    }
                }
                println!("--- {} results ---", results.len());
            }
        }

        Commands::Jstar { action } => match action {
            JStarAction::Compile {
                input,
                include,
                output,
                raw,
            } => {
                let mode = if raw { "raw" } else { "nlp" };
                if include.is_empty() {
                    println!(
                        "Compiling {} -> {} ({})",
                        input.display(),
                        output.display(),
                        mode
                    );
                    if raw {
                        morphlex::jstar::compile_file_raw(&input, &output)
                            .expect("JStar compilation failed");
                    } else {
                        morphlex::jstar::compile_file(&input, &output)
                            .expect("JStar compilation failed");
                    }
                } else {
                    let mut sources: Vec<PathBuf> = include;
                    sources.push(input.clone());
                    let paths: Vec<&std::path::Path> =
                        sources.iter().map(|p| p.as_path()).collect();
                    println!(
                        "Compiling {} files -> {} ({})",
                        paths.len(),
                        output.display(),
                        mode
                    );
                    morphlex::jstar::compile_multi(&paths, &output)
                        .expect("JStar compilation failed");
                }
                println!("Binary written to {}", output.display());
            }
            JStarAction::Kernel { input, output } => {
                println!(
                    "Compiling kernel {} -> {}",
                    input.display(),
                    output.display()
                );
                morphlex::jstar::compile_kernel_file(&input, &output)
                    .expect("Kernel compilation failed");
                println!("Kernel ELF written to {}", output.display());
                println!("Boot with: qemu-system-x86_64 -kernel {}", output.display());
            }
            JStarAction::Parse { input, text } => {
                let source = match (input, text) {
                    (Some(path), _) => {
                        std::fs::read_to_string(&path).expect("Failed to read source file")
                    }
                    (_, Some(text)) => text,
                    (None, None) => {
                        eprintln!("Error: provide --input <file> or --text <code>");
                        std::process::exit(1);
                    }
                };
                let (originals, lemmas, vectors) =
                    morphlex::jstar::tokenize_jstar(&source).expect("Tokenization failed");
                let program = morphlex::jstar::parser::parse(&originals, &lemmas, &vectors)
                    .expect("Parse failed");
                let typed =
                    morphlex::jstar::typechecker::check(&program).expect("Type check failed");
                println!("{:#?}", typed);
            }
        },

        Commands::Jsh { script } => match script {
            Some(path) => {
                morphlex::jsh::scripting::run_script(&path).expect("Script execution failed");
            }
            None => {
                morphlex::jsh::repl::run().expect("REPL error");
            }
        },

        Commands::Crawl {
            url,
            depth,
            delay,
            output,
            max_pages,
            user_agent,
        } => {
            let seed_url = url::Url::parse(&url).unwrap_or_else(|e| {
                eprintln!("Invalid URL '{}': {}", url, e);
                std::process::exit(1);
            });

            let max_pages_opt = if max_pages == 0 {
                None
            } else {
                Some(max_pages as usize)
            };

            let config = morphlex::crawler::CrawlConfig {
                seed_url,
                max_depth: depth,
                delay_ms: delay,
                output_dir: output,
                user_agent,
                max_pages: max_pages_opt,
            };

            let summary = morphlex::crawler::crawl(&config).expect("Crawl failed");
            println!(
                "Crawl complete: {} pages crawled, {} skipped -> {}",
                summary.pages_crawled,
                summary.pages_skipped,
                summary.output_dir.display(),
            );
        }

        Commands::Rr { action } => match action {
            RrAction::Deploy {
                mission,
                priority,
                db,
            } => {
                use morphlex::rr::{Mission, Priority, RRDatabase, SwarmOrchestrator};

                let priority = match priority.as_str() {
                    "urgent" => Priority::Urgent,
                    "priority" => Priority::Priority,
                    _ => Priority::Routine,
                };

                let mut orchestrator = SwarmOrchestrator::new();
                let mission_obj = Mission::new(&mission, priority);

                let swarm = orchestrator
                    .spawn_swarm(mission_obj)
                    .expect("Failed to spawn swarm");
                let swarm_id = swarm.id.clone();

                // Save to database
                let mut rr_db = RRDatabase::new();
                rr_db.record_swarm(&morphlex::rr::SwarmRecord::from(swarm));
                rr_db.save_to_path(&db).expect("Failed to save database");

                println!("Swarm deployed: {}", swarm_id);
                println!("  Mission: {}", mission);
                println!("  Priority: {:?}", priority);
                println!("  Agents: {}", swarm.agent_ids.len());
                println!("  Database: {}", db.display());
            }

            RrAction::Status { swarm_id, db } => {
                use morphlex::rr::RRDatabase;

                let rr_db = RRDatabase::load_from_path(&db).unwrap_or_else(|_| RRDatabase::new());

                if let Some(swarm) = rr_db.get_swarm(&swarm_id) {
                    println!("Swarm: {}", swarm.id);
                    println!("  Mission: {}", swarm.mission_id);
                    println!("  Status: {:?}", swarm.status);
                    println!("  Progress: {}%", swarm.progress);
                    println!("  Agents: {}", swarm.agent_ids.len());
                    println!("  Created: {}", swarm.created_at);
                } else {
                    eprintln!("Swarm not found: {}", swarm_id);
                    std::process::exit(1);
                }
            }

            RrAction::List { db } => {
                use morphlex::rr::RRDatabase;

                let rr_db = RRDatabase::load_from_path(&db).unwrap_or_else(|_| RRDatabase::new());
                let stats = rr_db.get_stats();

                println!("Active Swarms: {}", stats.active_swarms);
                for (id, swarm) in &rr_db.swarms {
                    if swarm.status == morphlex::rr::SwarmStatus::Active {
                        println!("  - {} ({}% complete)", id, swarm.progress);
                    }
                }

                if stats.active_swarms == 0 {
                    println!("  (no active swarms)");
                }
            }

            RrAction::Frago {
                swarm_id,
                content,
                db,
            } => {
                use morphlex::rr::{Frago, Priority, RRDatabase};

                let mut rr_db =
                    RRDatabase::load_from_path(&db).unwrap_or_else(|_| RRDatabase::new());

                if rr_db.get_swarm(&swarm_id).is_none() {
                    eprintln!("Swarm not found: {}", swarm_id);
                    std::process::exit(1);
                }

                let frago = Frago::new("commander".to_string(), swarm_id.clone(), content.clone());

                rr_db.record_communication(&morphlex::rr::CommunicationDBRecord {
                    id: frago.header.id.clone(),
                    swarm_id: Some(swarm_id.clone()),
                    from: frago.header.from.clone(),
                    to: frago.header.to.clone(),
                    comm_type: frago.header.comm_type,
                    content_summary: content,
                    timestamp: frago.header.timestamp,
                    priority: Priority::Priority,
                    full_content: serde_json::to_string(&frago).unwrap_or_default(),
                });

                rr_db.save_to_path(&db).expect("Failed to save database");
                println!("FRAGO issued to swarm {}", swarm_id);
            }

            RrAction::Disband { swarm_id, db } => {
                use morphlex::rr::RRDatabase;

                let mut rr_db =
                    RRDatabase::load_from_path(&db).unwrap_or_else(|_| RRDatabase::new());

                if let Some(swarm) = rr_db.get_swarm(&swarm_id) {
                    let mut swarm = swarm.clone();
                    swarm.status = morphlex::rr::SwarmStatus::Disbanded;
                    swarm.completed_at = Some(morphlex::rr::memory::now());
                    rr_db.record_swarm(&swarm);
                    rr_db.save_to_path(&db).expect("Failed to save database");
                    println!("Swarm {} disbanded", swarm_id);
                } else {
                    eprintln!("Swarm not found: {}", swarm_id);
                    std::process::exit(1);
                }
            }

            RrAction::CreateAgent {
                agent_type,
                name,
                db,
            } => {
                use morphlex::rr::{AgentType, RRDatabase, SwarmOrchestrator};

                let agent_type = match agent_type.as_str() {
                    "simple" => AgentType::Simple,
                    "multimodal" => AgentType::Multimodal,
                    "langchain" => AgentType::LangChain,
                    "guardian" => AgentType::Guardian,
                    "coding" => AgentType::Coding,
                    "analysis" => AgentType::DataAnalysis,
                    "search" => AgentType::SearchReplace,
                    "datamgmt" => AgentType::DataManagement,
                    "filtration" => AgentType::DataFiltration,
                    _ => {
                        eprintln!("Unknown agent type: {}", agent_type);
                        eprintln!(
                            "Valid types: simple, multimodal, langchain, guardian, coding, analysis, search, datamgmt, filtration"
                        );
                        std::process::exit(1);
                    }
                };

                let mut orchestrator = SwarmOrchestrator::new();
                let agent_id = orchestrator.create_agent(agent_type, name.clone());

                // Save to database
                let mut rr_db = RRDatabase::new();
                if let Some(record) = orchestrator.agents.get(&agent_id) {
                    rr_db.record_agent(&morphlex::rr::AgentDBRecord {
                        agent_id: record.agent_id.clone(),
                        name: record.name.clone(),
                        rank: record.rank,
                        mos: record.mos,
                        unit: None,
                        commander: None,
                        swarm_id: record.swarm_id.clone(),
                        status: record.status.clone(),
                        agent_type: record.agent_type,
                        created_at: morphlex::rr::memory::now(),
                        missions_completed: 0,
                        performance_score: 1.0,
                    });
                    rr_db.save_to_path(&db).expect("Failed to save database");
                }

                println!("Agent created: {}", agent_id);
                println!("  Type: {:?}", agent_type);
                println!("  Name: {}", name.unwrap_or_else(|| "Unnamed".to_string()));
                println!("  Database: {}", db.display());
            }

            RrAction::Stats { db } => {
                use morphlex::rr::RRDatabase;

                let rr_db = RRDatabase::load_from_path(&db).unwrap_or_else(|_| RRDatabase::new());
                let stats = rr_db.get_stats();
                println!("{}", stats);
            }
        },

        Commands::Daemon { action } => match action {
            DaemonAction::Integrity {
                monitor,
                check_interval,
            } => {
                use morphlex::rr::SystemIntegrityDaemon;

                let paths = if monitor.is_empty() {
                    vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))]
                } else {
                    monitor
                };

                println!("Starting System Integrity Daemon...");
                println!("  Monitoring: {:?}", paths);
                println!("  Check interval: {}s", check_interval);

                let daemon = SystemIntegrityDaemon::new(paths);
                daemon.start();

                println!("Daemon started. Press Ctrl+C to stop.");

                // Keep running
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(60));
                }
            }

            DaemonAction::Threat { scan_interval } => {
                use morphlex::rr::ThreatIntelligenceManager;

                println!("Starting Threat Intelligence Manager...");
                println!("  Scan interval: {}s", scan_interval);

                let mut manager = ThreatIntelligenceManager::new();
                manager.scan_interval = scan_interval;
                manager.start();

                println!("Manager started. Press Ctrl+C to stop.");

                // Keep running
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(60));
                }
            }

            DaemonAction::Maintenance { run, report } => {
                use morphlex::rr::MorphogeneticMaintainer;

                let mut maintainer = MorphogeneticMaintainer::new();

                if report {
                    // Just show last maintenance report
                    match maintainer.last_maintenance {
                        Some(ts) => println!("Last maintenance: {}", ts),
                        None => println!("No maintenance has been run yet."),
                    }
                } else if run || maintainer.should_run() {
                    println!("Running morphogenetic maintenance...");
                    match maintainer.run_maintenance() {
                        Ok(report) => {
                            println!("Maintenance completed in {}s", report.duration_seconds);
                            println!("Tasks completed:");
                            for task in &report.tasks_completed {
                                println!("  [DONE] {}", task);
                            }
                            if !report.optimizations.is_empty() {
                                println!("Optimizations:");
                                for opt in &report.optimizations {
                                    println!("  [DONE] {}", opt);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Maintenance failed: {}", e);
                            std::process::exit(1);
                        }
                    }
                } else {
                    println!("Maintenance not due. Next scheduled run: 4 AM daily");
                }
            }

            DaemonAction::StartAll { foreground } => {
                println!("Starting all Rational Reserve daemons...");

                if foreground {
                    println!("Running in foreground mode...");
                    // In foreground, we'd start threads for each daemon
                    // For now, just start them sequentially
                } else {
                    // Start in background
                    println!("Starting daemons in background...");
                }

                // Start integrity daemon
                {
                    use morphlex::rr::SystemIntegrityDaemon;
                    let daemon = SystemIntegrityDaemon::new(vec![
                        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
                    ]);
                    daemon.start();
                    println!("  [DONE] System Integrity Daemon started");
                }

                // Start threat manager
                {
                    use morphlex::rr::ThreatIntelligenceManager;
                    let manager = ThreatIntelligenceManager::new();
                    manager.start();
                    println!("  [DONE] Threat Intelligence Manager started");
                }

                println!("All daemons started.");
                println!("Use 'morphlex daemon maintenance --report' to check status.");
            }
        },

        Commands::Llm { action } => match action {
            LlmAction::Train {
                data,
                size,
                epochs,
                batch_size,
                lr,
                output,
            } => {
                use morphlex::llm::{
                    DataLoader, ModelConfig, MorphlexLLM, Trainer, TrainingConfig,
                };

                println!("=== MorphlexLLM Training ===");
                println!("Data: {}", data.display());
                println!("Model size: {}", size);
                println!("Epochs: {}", epochs);
                println!("Batch size: {}", batch_size);
                println!("Learning rate: {}", lr);
                println!("Output: {}", output);
                println!();

                // Create model config based on size
                let config = match size.as_str() {
                    "medium" => ModelConfig::medium(),
                    "large" => ModelConfig::large(),
                    _ => ModelConfig::small(),
                };

                println!("Model configuration:");
                println!("  d_model: {}", config.d_model);
                println!("  layers: {}", config.n_layers);
                println!("  heads: {}", config.n_heads);
                println!("  parameters: {}", config.param_count());
                println!();

                // Create model
                let model = MorphlexLLM::new(&config);

                // Load training data
                println!("Loading training data...");
                let mut dataloader = if data.extension().map_or(false, |e| e == "db")
                    || data.extension().map_or(false, |e| e == "enc")
                {
                    DataLoader::from_database(&data, batch_size, true, config.max_seq_len)
                        .expect("Failed to load database")
                } else {
                    DataLoader::from_text_file(&data, batch_size, true)
                        .expect("Failed to load text file")
                };

                println!("Loaded {} training samples", dataloader.num_samples());
                println!("Batches per epoch: {}", dataloader.num_batches());
                println!();

                // Create trainer
                let training_config = TrainingConfig {
                    epochs,
                    batch_size,
                    learning_rate: lr,
                    max_grad_norm: 1.0,
                    checkpoint_interval: 100,
                    log_interval: 10,
                    output_dir: output.clone(),
                };

                let mut trainer = Trainer::new(model, training_config);

                // Run training
                println!("Starting training...");
                let stats = trainer.train(&mut dataloader).expect("Training failed");

                println!();
                println!("=== Training Complete ===");
                println!("Epochs completed: {}", stats.epochs);
                println!("Total steps: {}", stats.steps);
                println!("Best loss: {:.4}", stats.best_loss);
                println!("Average loss: {:.4}", stats.avg_loss);
                println!("Tokens processed: {}", stats.tokens_processed);
                println!("Checkpoints saved to: {}", output);
            }

            LlmAction::Export {
                model,
                output,
                quantize,
            } => {
                use morphlex::llm::{export_to_gguf, MorphlexLLM};

                println!("Exporting model to GGUF format...");
                println!("Input: {}", model.display());
                println!("Output: {}", output.display());
                println!("Quantize: {}", quantize);

                let model = MorphlexLLM::load(&model).expect("Failed to load model");
                export_to_gguf(&model, &output, quantize).expect("Export failed");

                println!("Model exported successfully!");
            }

            LlmAction::Infer {
                model,
                prompt,
                max_tokens,
                temperature,
            } => {
                use morphlex::llm::MorphlexLLM;

                println!("Running inference...");
                println!("Model: {}", model.display());
                println!("Prompt: {}", prompt);
                println!("Max tokens: {}", max_tokens);
                println!("Temperature: {}", temperature);

                // Load model and run inference (placeholder)
                let _model = MorphlexLLM::load(&model).expect("Failed to load model");

                // In production: generate tokens
                println!();
                println!("Inference not yet fully implemented.");
                println!("Model loaded successfully - ready for generation.");
            }

            LlmAction::Info { model } => {
                use morphlex::llm::MorphlexLLM;

                println!("Loading model info...");
                let model = MorphlexLLM::load(&model).expect("Failed to load model");

                println!();
                println!("=== Model Information ===");
                println!("d_model: {}", model.config.d_model);
                println!("Layers: {}", model.config.n_layers);
                println!("Attention heads: {}", model.config.n_heads);
                println!("FFN dimension: {}", model.config.d_ff);
                println!("Vocabulary size: {}", model.config.vocab_size);
                println!("Max sequence length: {}", model.config.max_seq_len);
                println!("Total parameters: {}", model.param_count());
                println!("Role-aware attention: {}", model.config.use_role_attention);
                println!("Morphological gates: {}", model.config.use_morph_gates);
                println!("Lemma embeddings: {}", model.config.use_lemma_embeddings);
            }
        },

        Commands::Inspect { database, keys } => {
            let dk_bytes = std::fs::read(keys.join("dk.bin"))
                .expect("Failed to read dk.bin from key directory");

            let vk_path = keys.join("vk.bin");
            let vk_bytes = if vk_path.exists() {
                Some(std::fs::read(&vk_path).expect("Failed to read vk.bin"))
            } else {
                None
            };

            let slh_vk_path = keys.join("slh_vk.bin");
            let slh_vk_bytes = if slh_vk_path.exists() {
                Some(std::fs::read(&slh_vk_path).expect("Failed to read slh_vk.bin"))
            } else {
                None
            };

            let decrypted = morphlex::database::decrypt(
                &database,
                &dk_bytes,
                vk_bytes.as_deref(),
                slh_vk_bytes.as_deref(),
            )
            .expect("Decryption failed");

            let (lemmas, vectors) =
                morphlex::database::read_database(&decrypted).expect("Failed to parse database");

            println!("{} vectors in database", vectors.len());
            for (i, (lemma, tv)) in lemmas.iter().zip(vectors.iter()).take(20).enumerate() {
                let id = tv.id;
                let pos = tv.pos;
                let role = tv.role;
                println!(
                    "  [{:>5}] {:>20}  id={:>11}  pos={:?}  role={:?}",
                    i,
                    lemma,
                    id,
                    vectorizer::i8_to_pos(pos),
                    vectorizer::i8_to_role(role),
                );
            }
            if vectors.len() > 20 {
                println!("  ... and {} more", vectors.len() - 20);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn test_write_key_bundle_hardens_permissions() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("morphlex_keys_{unique}"));
        let bundle = morphlex::database::PqcKeyBundle {
            decapsulation_key: vec![1; 64],
            signing_key: vec![2; 32],
            verifying_key: vec![3; 16],
            slh_signing_key: vec![4; 128],
            slh_verifying_key: vec![5; 16],
        };

        write_key_bundle(&bundle, &dir);

        let dir_mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        let dk_mode = std::fs::metadata(dir.join("dk.bin"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let sk_mode = std::fs::metadata(dir.join("sk.bin"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let vk_mode = std::fs::metadata(dir.join("vk.bin"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let slh_sk_mode = std::fs::metadata(dir.join("slh_sk.bin"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let slh_vk_mode = std::fs::metadata(dir.join("slh_vk.bin"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;

        assert_eq!(dir_mode, 0o700);
        assert_eq!(dk_mode, 0o600);
        assert_eq!(sk_mode, 0o600);
        assert_eq!(vk_mode, 0o644);
        assert_eq!(slh_sk_mode, 0o600);
        assert_eq!(slh_vk_mode, 0o644);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
