import yaml
import json

glyphs = {"CICCA33O11ZI2C2H":['!', '"', '#', '$', '%', '&',
                              "'", '(', ')', '*', '+', ',',
                              '-', '.', '/', ':', ';', '<',
                              '=', '>', '?', '@', 'A', 'B',
                              'C', 'D', 'E', 'F', 'G', 'H',
                              'I', 'J', 'K', 'L', 'M', 'N',
                              'O', 'P', 'Q', 'R', 'S', 'T',
                              'U', 'V', 'W', 'X', 'Y', 'Z',
                              '[', ']', '^', '_', 'a', 'b',
                              'c', 'd', 'e', 'f', 'g', 'h'],
          "CICCA105S2B66D5C2B5D4B3A3D5C2B5D4B3A5B13B23B63A":['i', 'j', 'k', 'l', 'm', 'n',
                                                             'o', 'p', 'q', 'r', 's', 't',
                                                             'u', 'v', 'w', 'x', 'y', 'z',
                                                             '{', '}', '~', 'À', 'Á', 'Â',
                                                             'Ã', 'È', 'É', 'Ê', 'Ì', 'Í',
                                                             'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú',
                                                             'Ý', 'à', 'á', 'â', 'ã', 'è',
                                                             'é', 'ê', 'ì', 'í', 'ò', 'ó',
                                                             'ô', 'õ', 'ù', 'ú', 'ý', 'Ă',
                                                             'ă', 'Đ', 'đ', 'Ĩ', 'ĩ', 'Ũ'],
          "CICCA361A55B14B7408ZZE":['ũ', 'Ơ', 'ơ', 'Ư', 'ư', 'Ạ',
                                    'ạ', 'Ả', 'ả', 'Ấ', 'ấ', 'Ầ',
                                    'ầ', 'Ẩ', 'ẩ', 'Ẫ', 'ẫ', 'Ậ',
                                    'ậ', 'Ắ', 'ắ', 'Ằ', 'ằ', 'Ẳ',
                                    'ẳ', 'Ẵ', 'ẵ', 'Ặ', 'ặ', 'Ẹ',
                                    'ẹ', 'Ẻ', 'ẻ', 'Ẽ', 'ẽ', 'Ế',
                                    'ế', 'Ề', 'ề', 'Ể', 'ể', 'Ễ',
                                    'ễ', 'Ệ', 'ệ', 'Ỉ', 'ỉ', 'Ị',
                                    'ị', 'Ọ', 'ọ', 'Ỏ', 'ỏ', 'Ố',
                                    'ố', 'Ồ', 'ồ', 'Ổ', 'ổ', 'Ỗ'],
          "CICCA7895ZJ":['ỗ', 'Ộ', 'ộ', 'Ớ', 'ớ', 'Ờ',
                         'ờ', 'Ở', 'ở', 'Ỡ', 'ỡ', 'Ợ',
                         'ợ', 'Ụ', 'ụ', 'Ủ', 'ủ', 'Ứ',
                         'ứ', 'Ừ', 'ừ', 'Ử', 'ử', 'Ữ',
                         'ữ', 'Ự', 'ự', 'Ỳ', 'ỳ', 'Ỵ',
                         'ỵ', 'Ỷ', 'ỷ', 'Ỹ', 'ỹ']}

config_font_creation = {"props": {"ascent": 800,
                                  "descent": 200,
                                  "em": 1000,
                                  "encoding": "UnicodeFull",
                                  "lang": "English (US)",
                                  "filename": "MyFont",
                                  "style": "Regular"},
                        "sfnt_names": {"Copyright": "Copyright (c) 2021 by Nobody",
                                       "Family": "MyFont",
                                       "SubFamily": "Regular",
                                       "UniqueID": "MyFont 2021-02-04",
                                       "Fullname": "MyFont Regular",
                                       "Version": "Version 1.0",
                                       "PostScriptName": "MyFont-Regular"},
                        "glyphs": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 192, 193, 194, 195, 200, 201, 202, 204, 205, 210, 211, 212, 213, 217, 218, 221, 224, 225, 226, 227, 232, 233, 234, 236, 237, 242, 243, 244, 245, 249, 250, 253, 258, 259, 272, 273, 296, 297, 360, 361, 416, 417, 431, 432, 7840, 7841, 7842, 7843, 7844, 7845, 7846, 7847, 7848, 7849, 7850, 7851, 7852, 7853, 7854, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7870, 7871, 7872, 7873, 7874, 7875, 7876, 7877, 7878, 7879, 7880, 7881, 7882, 7883, 7884, 7885, 7886, 7887, 7888, 7889, 7890, 7891, 7892, 7893, 7894, 7895, 7896, 7897, 7898, 7899, 7900, 7901, 7902, 7903, 7904, 7905, 7906, 7907, 7908, 7909, 7910, 7911, 7912, 7913, 7914, 7915, 7916, 7917, 7918, 7919, 7920, 7921, 7922, 7923, 7924, 7925, 7926, 7927, 7928, 7929]}


with open('config/glyphs.yaml', 'w', encoding='utf-8') as outfile:
    yaml.dump(glyphs, outfile, sort_keys=False, allow_unicode=True)

json_object = json.dumps(config_font_creation, indent=4)
with open("config/Quan_font_config.json", "w") as outfile:
    outfile.write(json_object)

