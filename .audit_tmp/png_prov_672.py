import struct, sys, os, datetime

def png_info(path):
    with open(path, 'rb') as f:
        data = f.read(64 * 1024)
    # tEXt chunks
    texts = []
    i = 8
    while i + 8 <= len(data):
        n = struct.unpack('>I', data[i:i+4])[0]
        typ = data[i+4:i+8]
        payload = data[i+8:i+8+n]
        if typ == b'tEXt':
            texts.append(payload.decode('latin-1'))
        if typ in (b'IDAT', b'IEND'):
            break
        i += 12 + n
    st = os.stat(path)
    print('file      :', os.path.basename(path))
    print('bytes     :', st.st_size)
    print('mtime     :', datetime.datetime.fromtimestamp(st.st_mtime))
    print('tEXt      :', texts)

for p in sys.argv[1:]:
    print('=' * 70)
    png_info(p)
