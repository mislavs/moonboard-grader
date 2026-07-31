"""Create an immutable experiment cohort manifest."""

from pathlib import Path

from src.experiment import load_cohort, resolve_experiment_config
from src.experiment_manifest import create_manifest_document, write_manifest

from .utils import console, print_completion_message, print_section_header


def setup_create_manifest_parser(subparsers):
    parser = subparsers.add_parser(
        "create-manifest",
        help="Freeze a filtered cohort and its outer/inner split memberships",
    )
    parser.add_argument("--config", default="config.yaml", help="Official experiment YAML")
    parser.add_argument("--output", required=True, help="New manifest JSON path")
    parser.set_defaults(func=create_manifest_command)
    return parser


def create_manifest_command(args):
    print_section_header("CREATE FROZEN EXPERIMENT MANIFEST")
    config, data_path = resolve_experiment_config(args.config)
    records, identity = load_cohort(config, data_path)
    output = Path(args.output)
    manifest = create_manifest_document(config, records, identity, output.stem)
    write_manifest(output, manifest)
    console.print(f"Manifest: {output}")
    console.print(f"Cohort problems: {len(records)}")
    console.print(f"Development: {len(manifest['split']['development_ids'])}")
    console.print(f"Locked test: {len(manifest['split']['test_ids'])}")
    console.print(f"SHA-256: {manifest['manifest_sha256']}")
    print_completion_message("Frozen manifest created")

