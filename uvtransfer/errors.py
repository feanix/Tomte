"""Exceptions raised by the uvtransfer engine."""


class TransferError(Exception):
    """A problem with the inputs (files, meshes or settings) that the user needs to fix.

    Messages are written for end users and can be shown as they are.
    """


class Cancelled(Exception):
    """Raised inside a transfer when its cancel callback returns True."""
