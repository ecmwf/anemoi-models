

import warnings


def shouldhasattr(obj: type, attr: str, description: str = None) -> bool:
    """
    Check if an object has an attribute and raise a warning if it doesn't.

    Parameters
    ----------
    obj : type
        Object to check.
    attr : str
        Attribute to check for existence.
    description : str, optional
        Description to include in warning if not found, by default None

    Returns
    -------
    bool
        Whether the object has the attribute or not.

    """    
    if not hasattr(obj, attr):
        description = "\n" + description or ""
        warnings.warn(f"{type(obj).__name__} should have attribute {attr!r} but doesn't. If loading an older checkpoint, check the versions.{description}", FutureWarning)
        return False
    return True
