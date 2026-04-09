use std::fmt::Write;

use crate::snake::Pos;
use crate::stage::StageKind;

const CANVAS: i32 = 360;
const GRID: i32 = 20;
const CELL: i32 = 18;

pub struct ScreenshotData {
    pub body: Vec<Pos>,
    pub food: (i32, i32),
    pub obstacles: Vec<(i32, i32)>,
    pub score: u32,
    pub gen: u32,
    pub snake_id: usize,
    pub stage_kind: StageKind,
}

/// Renders a 360x360 game screenshot as PNG bytes.
pub fn render_screenshot(data: &ScreenshotData) -> Option<Vec<u8>> {
    let svg = build_svg(data);

    let mut opt = usvg::Options {
        font_family: "monospace".into(),
        ..Default::default()
    };
    opt.fontdb_mut().load_system_fonts();

    let tree = usvg::Tree::from_str(&svg, &opt).ok()?;

    let size = tree.size();
    let mut pixmap = tiny_skia::Pixmap::new(size.width() as u32, size.height() as u32)?;
    resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());

    pixmap.encode_png().ok()
}

/// Helper: write an SVG rect element
fn rect(svg: &mut String, x: i32, y: i32, w: i32, h: i32, fill: &str, extra: &str) {
    let _ = write!(svg, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" {}/>", x, y, w, h, fill, extra);
}

/// Helper: write an SVG line element
fn line(svg: &mut String, x1: i32, y1: i32, x2: i32, y2: i32, stroke: &str, sw: &str, extra: &str) {
    let _ = write!(svg, "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\" {}/>", x1, y1, x2, y2, stroke, sw, extra);
}

/// Helper: write an SVG text element
fn text(svg: &mut String, x: i32, y: i32, size: i32, fill: &str, content: &str, extra: &str) {
    let _ = write!(svg, "<text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"{}\" fill=\"{}\" text-anchor=\"middle\" dominant-baseline=\"central\" {}>{}</text>", x, y, size, fill, extra, content);
}

fn build_svg(data: &ScreenshotData) -> String {
    let mut svg = String::with_capacity(16384);
    let _ = write!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">", CANVAS, CANVAS, CANVAS, CANVAS);

    let is_warehouse = matches!(data.stage_kind, StageKind::Warehouse);

    if is_warehouse {
        write_warehouse_bg(&mut svg);
    } else {
        write_classic_bg(&mut svg);
    }

    write_obstacles(&mut svg, &data.obstacles);
    write_food(&mut svg, data.food, is_warehouse);
    write_snake(&mut svg, &data.body, is_warehouse);
    write_banner(&mut svg, data, is_warehouse);

    svg.push_str("</svg>");
    svg
}

fn write_classic_bg(svg: &mut String) {
    rect(svg, 0, 0, CANVAS, CANVAS, "#0a0a0a", "");
    for i in 0..=GRID {
        let pos = i * CELL;
        line(svg, pos, 0, pos, CANVAS, "#151515", "0.5", "");
        line(svg, 0, pos, CANVAS, pos, "#151515", "0.5", "");
    }
}

fn write_warehouse_bg(svg: &mut String) {
    rect(svg, 0, 0, CANVAS, CANVAS, "#d4d0c8", "");
    for i in 0..=GRID {
        let pos = i * CELL;
        line(svg, pos, 0, pos, CANVAS, "#c0bbb2", "0.5", "");
        line(svg, 0, pos, CANVAS, pos, "#c0bbb2", "0.5", "");
    }
    // Yellow safety border
    let _ = write!(svg, "<rect x=\"1.5\" y=\"1.5\" width=\"357\" height=\"357\" fill=\"none\" stroke=\"#e8b830\" stroke-width=\"3\"/>");
    // Dashed lane guides
    line(svg, 0, 180, 360, 180, "#daa520", "1.5", "stroke-dasharray=\"8,6\"");
    line(svg, 180, 0, 180, 360, "#daa520", "1.5", "stroke-dasharray=\"8,6\"");
    // Green pathway strips
    rect(svg, 0, 0, CANVAS, CELL, "rgba(80,180,80,0.12)", "");
    rect(svg, 0, CANVAS - CELL, CANVAS, CELL, "rgba(80,180,80,0.12)", "");
    rect(svg, 0, 0, CELL, CANVAS, "rgba(80,180,80,0.12)", "");
    rect(svg, CANVAS - CELL, 0, CELL, CANVAS, "rgba(80,180,80,0.12)", "");
}

fn write_obstacles(svg: &mut String, obstacles: &[(i32, i32)]) {
    for &(ox, oy) in obstacles {
        let x = ox * CELL;
        let y = oy * CELL;
        rect(svg, x + 1, y + 1, CELL, CELL, "rgba(0,0,0,0.1)", "rx=\"2\"");
        rect(svg, x, y, CELL, CELL, "#c48a3c", "rx=\"2\" stroke=\"#a06a20\" stroke-width=\"0.6\"");
        rect(svg, x + 1, y + 1, CELL - 2, 3, "#d4a050", "");
        // Mini cargo boxes
        let boxes: &[(&str, i32, i32)] = &[("#6aaa4a", 2, 5), ("#4488cc", 9, 5), ("#cc6644", 2, 10), ("#8866aa", 9, 10)];
        for &(color, dx, dy) in boxes {
            let _ = write!(svg, "<rect x=\"{}\" y=\"{}\" width=\"5\" height=\"4\" rx=\"0.5\" fill=\"{}\"/>", x + dx, y + dy, color);
        }
    }
}

fn write_food(svg: &mut String, food: (i32, i32), is_warehouse: bool) {
    let cx = food.0 * CELL + CELL / 2;
    let cy = food.1 * CELL + CELL / 2;

    if is_warehouse {
        let x = food.0 * CELL + 2;
        let y = food.1 * CELL + 2;
        let s = CELL - 4;
        rect(svg, x, y, s, s, "#e8a040", "rx=\"2\" stroke=\"#b07020\" stroke-width=\"0.7\"");
        // Packing tape cross
        line(svg, x + s / 2, y + 1, x + s / 2, y + s - 1, "#c87828", "1.5", "");
        line(svg, x + 1, y + s / 2, x + s - 1, y + s / 2, "#c87828", "1.5", "");
    } else {
        let _ = write!(svg, "<circle cx=\"{}\" cy=\"{}\" r=\"3.6\" fill=\"#f33\"/>", cx, cy);
    }
}

fn write_snake(svg: &mut String, body: &[Pos], is_warehouse: bool) {
    let len = body.len() as f32;
    if body.is_empty() {
        return;
    }

    for i in (0..body.len()).rev() {
        let b = &body[i];
        let is_head = i == 0;
        let t = 1.0 - i as f32 / len;
        let opacity = 0.3 + t * 0.7;
        let pad = if is_head { 1 } else { 2 };
        let x = b.x * CELL + pad;
        let y = b.y * CELL + pad;
        let size = CELL - pad * 2;

        if is_warehouse {
            if is_head {
                // AMR chassis
                let _ = write!(svg, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"5\" fill=\"#2888c8\" stroke=\"#1868a0\" stroke-width=\"1\"/>", x, y, size, size);
                // Top plate
                let _ = write!(svg, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"3\" fill=\"#40a8e8\"/>", x + 2, y + 2, size - 4, size - 4);
                // Lidar dome
                let cx = b.x * CELL + CELL / 2;
                let cy = b.y * CELL + CELL / 2;
                let _ = write!(svg, "<circle cx=\"{}\" cy=\"{}\" r=\"3\" fill=\"#e0f0ff\" stroke=\"#60b0e0\" stroke-width=\"0.5\"/>", cx, cy);
                let _ = write!(svg, "<circle cx=\"{}\" cy=\"{}\" r=\"1.2\" fill=\"#2080c0\"/>", cx, cy);
            } else {
                // Cart
                let _ = write!(svg, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"3\" fill=\"#b8c0c8\" opacity=\"{:.2}\" stroke=\"#8898a8\" stroke-width=\"0.8\"/>", x, y, size, size, opacity);
                let cargo_colors = ["#e8a040", "#6aaa4a", "#4488cc", "#cc6644"];
                let color = cargo_colors[i % cargo_colors.len()];
                let _ = write!(svg, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"1\" fill=\"{}\" opacity=\"{:.2}\"/>", x + 3, y + 3, size - 6, size - 6, color, opacity);
            }
        } else if is_head {
            let _ = write!(svg, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"2\" fill=\"#00ff88\"/>", x, y, size, size);
        } else {
            let _ = write!(svg, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"2\" fill=\"#008c50\" opacity=\"{:.2}\"/>", x, y, size, size, opacity);
        }
    }
}

fn write_banner(svg: &mut String, data: &ScreenshotData, is_warehouse: bool) {
    let by = 270;

    // Dark background
    rect(svg, 0, by, CANVAS, 90, "rgba(0,0,0,0.85)", "");
    // Gold separator
    rect(svg, 0, by - 1, CANVAS, 2, "#ffd700", "");

    // Title
    text(svg, 180, by + 20, 16, "#ffd700", "NEW RECORD!", "font-weight=\"bold\"");

    // Score
    let score_str = format!("Score: {}", data.score);
    text(svg, 180, by + 44, 20, "#ffffff", &score_str, "font-weight=\"bold\"");

    // Stats
    let prefix = if is_warehouse { "AMR" } else { "Snake" };
    let stage = match data.stage_kind {
        StageKind::Classic => "Classic",
        StageKind::Warehouse => "Warehouse",
        StageKind::Mixed => "Mixed",
    };
    let stats = format!("{} #{}  |  Gen {}  |  {}", prefix, data.snake_id + 1, data.gen, stage);
    text(svg, 180, by + 64, 9, "#aaaaaa", &stats, "");

    // Timestamp
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    text(svg, 180, by + 80, 8, "#666666", &ts, "");
}
