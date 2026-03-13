import struct
from struct import unpack
from PIL import Image, ImageDraw

# Function from https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py
def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'country_code': country_code,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


# Function from https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py
def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


# Load all drawings from the binary file
def load_all_drawings(filename):
    drawings = []
    for drawing in unpack_drawings(filename):
        drawings.append(drawing)
    return drawings

# Draw an image from the stroke data
def draw_to_image(strokes, size = 256, img = None, offset = (0, 0), animated = False):
    animation_offset = 1    # Minimum 1
    animation_scale = 2     # 1, higher numbers flatten out the variation
    original_size = 256
    if img is None:
        img = Image.new('L', (size, size), color = 255 if animated else 0)  # Create a black canvas
    draw = ImageDraw.Draw(img)
    # Total number of line segments in all strokes
    total_segments = sum(len(stroke[0]) - 1 for stroke in strokes)
    segment_num = total_segments - 1
    # Draw each stroke in reverse order so earlier strokes are on top
    for stroke in reversed(strokes):
        x, y = stroke
        points = list(zip(x, y))
        points = [(offset[0] + int(px * size / original_size), offset[1] + int(py * size / original_size)) for px, py in points]
        if not animated:
            draw.line(points, fill=255, width=2)
        else:
            # Draw each segment in a different shade of gray to create an animation effect
            for i in range(len(points) - 1, 1, -1):
                # Calculate level 1-254 directly from segment number, and scale if > 254 segments
                shade = int(animation_offset + (animation_scale * segment_num / total_segments) * (254 - animation_offset)) if int(animation_offset + animation_scale * total_segments) >= 254 else int(animation_offset + animation_scale * segment_num)
                draw.line([points[i], points[i - 1]], fill=shade, width=2)
                segment_num -= 1
    return img

# Draw a tiled set of images
def draw_tiled(stroke_data, pos = 0, count = (1, 1), size = 32, animated = False):
    img = Image.new('L', (count[0] * size, count[1] * size), color = 255 if animated else 0)  # Create a black canvas
    for y in range(count[1]):
        for x in range(count[0]):
            index = pos + y * count[1] + x
            if index < len(stroke_data):
                strokes = stroke_data[index]['image']
                offset = (x * size, y * size)
                draw_to_image(strokes, size=size, img=img, offset=offset, animated=animated)
    return img


def create_tiles(filename, count=(32, 32), size=32):
    print(f'Loading drawings from {filename}...')
    drawings = load_all_drawings(filename)
    print(f'Total drawings loaded: {len(drawings)}')

    base_name = filename.split('/')[-1].split('.')[0]
    if base_name.startswith('full_binary_'):
        base_name = base_name[len('full_binary_'):]
    if not base_name:
        base_name = 'output'

    tiles_per_image = count[0] * count[1]
    print(f'Creating tiles {count} => {tiles_per_image} drawings per image...')
    image_resolution = (size * count[0], size * count[1])
    print(f'Image resolution, each tile {size}x{size} => {image_resolution[0]}x{image_resolution[1]}')

    image_count = len(drawings) // tiles_per_image
    print(f'Creating {image_count} tiled images...')

    for i in range(image_count):
        out_filename = f'{base_name}_{i:04d}.png'
        print(f'Creating: {out_filename} with drawings {i * tiles_per_image} to {(i + 1) * tiles_per_image - 1}...')
        img = draw_tiled(drawings, i * tiles_per_image, count, size, True)
        img.save(out_filename)
    
    skipped_drawings = len(drawings) % tiles_per_image
    if skipped_drawings > 0:
        print(f'Note: Skipped last {skipped_drawings} drawings that did not fit into a full tile set.')
    
    print('Done!')


if __name__ == "__main__":
    filename = 'data/full_binary_cup.bin'
    create_tiles(filename)
