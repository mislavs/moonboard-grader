"""Hard deprecation for the leakage-prone legacy training command."""


def setup_train_parser(subparsers):
    parser = subparsers.add_parser(
        "train",
        help="Deprecated; use the frozen experiment workflow",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Retained only to provide migration guidance",
    )
    parser.set_defaults(func=deprecated_train_command)
    return parser


def deprecated_train_command(args):
    raise RuntimeError(
        "The legacy train command is disabled because it evaluated the test set on every run. "
        "Use create-manifest, cross-validate, refit, and evaluate instead."
    )


# Preserve the import name while making programmatic calls fail just like the CLI.
train_command = deprecated_train_command
