import argparse
from pathlib import Path
from typing import Optional, Set


def should_skip(name: str, skip_hidden: bool) -> bool:
    return skip_hidden and name.startswith('.')


def iter_entries(path: Path, skip_hidden: bool, exclude_paths: Optional[Set[Path]] = None):
    entries = []
    try:
        for entry in path.iterdir():
            if should_skip(entry.name, skip_hidden):
                continue
            if exclude_paths and entry.resolve() in exclude_paths:
                continue
            entries.append(entry)
    except PermissionError:
        return []
    return sorted(entries, key=lambda p: (not p.is_dir(), p.name.lower()))


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def build_tree_lines(
    root: Path,
    max_depth: Optional[int],
    skip_hidden: bool,
    show_files: bool,
    exclude_paths: Optional[Set[Path]] = None,
) -> list:
    lines = [f"Root: {root.resolve()}"]

    def walk(current: Path, prefix: str = "", depth: int = 0):
        if max_depth is not None and depth >= max_depth:
            return

        entries = iter_entries(current, skip_hidden, exclude_paths)
        if not show_files:
            entries = [entry for entry in entries if entry.is_dir()]

        for index, entry in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            lines.append(f"{prefix}{connector}{entry.name}/" if entry.is_dir() else f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "|   "
                walk(entry, prefix + extension, depth + 1)

    walk(root)
    return lines


def write_output(lines: list, output_path: Path):
    output_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_output_path(root: Path, output_arg: Optional[str]) -> Path:
    if output_arg is None:
        return (root / "dataset_tree.txt").resolve()

    output_path = Path(output_arg)
    if output_path.is_absolute():
        return output_path.resolve()
    return (root / output_path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Export dataset folder structure to a text file.")
    parser.add_argument("root", nargs="?", default=".", help="Dataset root directory, default is current directory")
    parser.add_argument("-o", "--output", default=None, help="Output text file path, default is <root>/dataset_tree.txt")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum directory depth to display, default 3")
    parser.add_argument("--include-files", action="store_true", help="Include files in the output")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files and directories")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    output_path = resolve_output_path(root, args.output)
    exclude_paths = {output_path}
    script_path = Path(__file__).resolve()
    if is_relative_to(script_path, root):
        exclude_paths.add(script_path)

    lines = build_tree_lines(
        root=root,
        max_depth=args.max_depth,
        skip_hidden=not args.include_hidden,
        show_files=args.include_files,
        exclude_paths=exclude_paths,
    )
    write_output(lines, output_path)

    print(f"Saved folder structure to: {output_path}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
