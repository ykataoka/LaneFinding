class BreakIt(Exception):
    pass


def mul(x, y):
    print(x*y)
    if x*y > 50:
        return 'error'
    else:
        return 'okay'
