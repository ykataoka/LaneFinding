from hoge import mul


class BreakIt(Exception):
    pass

for i in range(0, 5, 1):
    try : 
        for x in range(10):
            for y in range(10):
                out = mul(x, y)
                if out == 'error':
                    raise BreakIt
    except BreakIt:
        print('done' + str(i))
