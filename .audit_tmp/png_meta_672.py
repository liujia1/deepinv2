# -*- coding: utf-8 -*-
"""读取PNG的tEXt元数据，判断图片是由哪个字体渲染（本地 or Colab产物）"""
import zlib, struct, sys, os

def chunks(path):
    with open(path, 'rb') as f:
        assert f.read(8) == b'\x89PNG\r\n\x1a\n'
        out = []
        while True:
            ln = f.read(4)
            if len(ln) < 4:
                break
            n = struct.unpack('>I', ln)[0]
            typ = f.read(4)
            data = f.read(n)
            f.read(4)
            out.append((typ, data))
            if typ == b'IDAT':
                break
        return out

for p in sys.argv[1:]:
    print('=' * 70)
    print(p, os.path.getsize(p), 'bytes', 'mtime', os.path.getmtime(p))
    for typ, data in chunks(p):
        if typ in (b'tEXt', b'iTXt'):
            try:
                print('  ', typ.decode(), data[:200])
            except Exception:
                print('  ', typ, '<bin>')
        else:
            print('  ', typ.decode(errors='replace'), len(data))
