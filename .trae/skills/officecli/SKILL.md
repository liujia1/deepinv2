---
name: "officecli"
description: "Create, analyze, and modify Office documents (.docx, .xlsx, .pptx) without Office installation. Invoke when user wants to create/edit Word/Excel/PPT files, check formatting, add charts, or generate reports/slides."
---

# officecli

AI-friendly CLI for .docx, .xlsx, .pptx. Single binary, no dependencies, no Office installation needed.

## Installation Status

- **Version**: 1.0.135
- **Path**: `C:\Users\admin\AppData\Local\OfficeCLI\officecli.exe`
- **Note**: Restart terminal before first use, then run `officecli --version`

## Strategy

**L1 (read) → L2 (DOM edit) → L3 (raw XML)**. Always prefer higher layers. Add `--json` for structured output.

## Help System (IMPORTANT)

**When unsure about property names, value formats, or command syntax, ALWAYS run help instead of guessing.**

```powershell
officecli help                                  # All commands + global options
officecli help docx                             # List all docx elements
officecli help docx paragraph                   # Full schema: properties, aliases, examples
officecli help xlsx cell                        # Excel cell properties
officecli help pptx slide                       # PowerPoint slide properties
```

---

## Quick Start

**PPT (PowerPoint):**
```powershell
officecli create slides.pptx
officecli add slides.pptx / --type slide --prop title="Q4 Report" --prop background=1A1A2E
officecli add slides.pptx '/slide[1]' --type shape --prop text="Revenue grew 25%" --prop x=2cm --prop y=5cm --prop font=Arial --prop size=24 --prop color=FFFFFF
```

**Word (.docx):**
```powershell
officecli create report.docx
officecli add report.docx /body --type paragraph --prop text="Executive Summary" --prop style=Heading1
officecli add report.docx /body --type paragraph --prop text="Revenue increased by 25% year-over-year."
```

**Excel (.xlsx):**
```powershell
officecli create data.xlsx
officecli set data.xlsx /Sheet1/A1 --prop value="Name" --prop bold=true
officecli set data.xlsx /Sheet1/A2 --prop value="Alice"
```

---

## L1: Create, Read & Inspect

```powershell
officecli create <file>               # Create blank .docx/.xlsx/.pptx
officecli view <file> <mode>          # outline | stats | issues | text | annotated | html
officecli get <file> <path> --depth N # Get a node and its children [--json]
officecli query <file> <selector>     # CSS-like query
officecli validate <file>             # Validate against OpenXML schema
```

### view modes

| Mode | Description |
|------|-------------|
| `outline` | Document structure |
| `stats` | Statistics (pages, words, shapes) |
| `issues` | Formatting/content/structure problems |
| `text` | Plain text extraction |
| `html` | Static HTML snapshot |

---

## L2: DOM Operations

### set — modify properties

```powershell
officecli set <file> <path> --prop key=value [--prop ...]
```

**Value formats:**
- Colors: `FF0000`, `#FF0000`, `red`, `rgb(255,0,0)`
- Spacing: `12pt`, `0.5cm`, `1.5x`
- Dimensions: `2.54cm`, `1in`, `72pt`

### find — format or replace matched text

```powershell
# Format matched text
officecli set doc.docx '/body/p[1]' --find weather --prop bold=true --prop color=red

# Replace text (whole document)
officecli set doc.docx / --find draft --replace final
```

### add — add elements

```powershell
officecli add <file> <parent> --type <type> [--prop ...]
officecli add <file> <parent> --type <type> --after <path> [--prop ...]   # insert after
officecli add <file> <parent> --type <type> --before <path> [--prop ...]  # insert before
```

**Element types:**
- **pptx**: slide, shape, picture, chart, table, connector, group, video, animation, transition
- **docx**: paragraph, run, table, row, cell, image, header, footer, section, bookmark, comment
- **xlsx**: sheet, row, col, cell, chart, image, pivottable, validation, autofilter

### remove, move, swap

```powershell
officecli remove <file> '/body/p[4]'
officecli move <file> <path> [--to <parent>] [--index N]
officecli swap <file> <path1> <path2>
```

---

## Watch & Interactive Selection

Live HTML preview that auto-refreshes on every file change.

```powershell
officecli watch <file> [--port N]      # Start preview server (default 26315)
officecli get <file> selected [--json] # Read what user clicked in browser
officecli unwatch <file>               # Stop
```

---

## Specialized Skills

Load specialized skills for specific document types:

```powershell
officecli load_skill pitch-deck        # Fundraising decks
officecli load_skill academic-paper    # Journal/conference papers
officecli load_skill financial-model   # Financial models
officecli load_skill data-dashboard    # Analytics dashboards
```

---

## Common Pitfalls

| Pitfall | Correct Approach |
|---------|-----------------|
| `--name "foo"` | Use `--prop name="foo"` |
| Unquoted `[N]` paths | Always quote: `'/slide[1]'` |
| Guessing property names | Run `officecli help <format> <element>` |
| Modifying an open file | Close the file in PowerPoint/Word first |

---

## Notes

- Paths are **1-based**: `'/body/p[3]'` = third paragraph
- Use `--json` for structured output when parsing results
- After modifications, verify with `validate` and/or `view issues`