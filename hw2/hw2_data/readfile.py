import sys

print(sys.argv[1])
f = open(sys.argv[1], 'r')
lines = f.read(16)
print(lines)

