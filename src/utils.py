
try:
    from IPython.display import display  # noqa: F401
except Exception:
    def display(*args, **kwargs):
        for a in args:
            print(a)
