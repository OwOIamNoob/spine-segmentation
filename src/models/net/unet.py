def toy(a:int=1, b:int=2):
    print(a, b)

args = {'a': 3, 'b': 5}
# print(**args)
toy(**args)