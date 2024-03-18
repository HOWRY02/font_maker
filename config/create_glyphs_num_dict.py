import yaml

with open('config/glyphs.yaml') as yaml_file:
        glyphs = yaml.safe_load(yaml_file)

# for char in glyphs['CICCA7895ZJ']:
#     print(ord(char))

glyphs_num = []
for value in glyphs.values():
    # print(value)
    for k in value:
        glyphs_num.append(ord(k))

print(glyphs_num)
print(len(glyphs_num))
