use std::path::PathBuf;
use std::sync::OnceLock;

static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Returns the directory where all persistent data is stored:
/// checkpoint.json, leaderboard.json, screenshots/
///
/// Resolution order:
/// 1. If SNAKE_AI_DATA_DIR env var is set, use that
/// 2. If running in a directory with Cargo.toml (dev mode), use that directory
/// 3. Otherwise, use platform-standard app data:
///    - macOS: ~/Library/Application Support/com.snake-ai.app/
///    - Linux: ~/.local/share/snake-ai/
///    - Windows: %APPDATA%/snake-ai/
pub fn data_dir() -> &'static PathBuf {
    DATA_DIR.get_or_init(|| {
        // 1. Explicit env var
        if let Ok(dir) = std::env::var("SNAKE_AI_DATA_DIR") {
            let p = PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&p);
            return p;
        }

        // 2. Dev mode: current dir has Cargo.toml
        let cwd = std::env::current_dir().unwrap_or_default();
        if cwd.join("Cargo.toml").exists() {
            return cwd;
        }

        // 3. Platform app data directory
        let base = if cfg!(target_os = "macos") {
            dirs_fallback_mac()
        } else if cfg!(target_os = "windows") {
            std::env::var("APPDATA")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("snake-ai")
        } else {
            // Linux / other
            std::env::var("XDG_DATA_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                    PathBuf::from(home).join(".local/share")
                })
                .join("snake-ai")
        };

        let _ = std::fs::create_dir_all(&base);
        base
    })
}

fn dirs_fallback_mac() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let p = PathBuf::from(home)
        .join("Library/Application Support/com.snake-ai.app");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Resolve a filename relative to the data directory
pub fn data_path(filename: &str) -> PathBuf {
    data_dir().join(filename)
}
