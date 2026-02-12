import numpy as np
import sys
import os
from pathlib import Path

INPUT_BITS = 16
EFFECTIVE_BITS = 14
OUTPUT_BITS = 20
LUT_SIZE = 16384

FORWARD_PWL = np.array([
    0, 118, 236, 353, 471, 589, 707, 825, 942, 1060, 1178, 1296, 1413, 1531, 
    1649, 1767, 1885, 2002, 2120, 2238, 2356, 2474, 2591, 2709, 2827, 2945, 
    3062, 3180, 3298, 3416, 3534, 3651, 3769, 3887, 4005, 4123, 4240, 4358, 
    4476, 4594, 4711, 4829, 4947, 5065, 5183, 5300, 5415, 5499, 5583, 5668, 
    5752, 5836, 5921, 6005, 6089, 6174, 6258, 6342, 6427, 6511, 6595, 6680, 
    6764, 6848, 6933, 7017, 7101, 7186, 7270, 7354, 7439, 7523, 7607, 7692, 
    7776, 7860, 7945, 8029, 8113, 8198, 8282, 8366, 8450, 8535, 8619, 8703, 
    8788, 8872, 8956, 9041, 9125, 9209, 9294, 9378, 9462, 9547, 9631, 9715, 
    9800, 9884, 9968, 10053, 10137, 10221, 10306, 10390, 10474, 10559, 10643, 
    10727, 10812, 10896, 10980, 11065, 11149, 11233, 11318, 11402, 11486, 
    11570, 11655, 11739, 11823, 11908, 11992, 12076, 12161, 12245, 12303, 
    12335, 12368, 12400, 12432, 12464, 12496, 12528, 12560, 12592, 12625, 
    12657, 12689, 12721, 12753, 12785, 12817, 12849, 12882, 12914, 12946, 
    12978, 13010, 13042, 13074, 13106, 13139, 13171, 13203, 13235, 13267, 
    13299, 13331, 13363, 13396, 13428, 13460, 13492, 13524, 13556, 13588, 
    13620, 13652, 13685, 13717, 13749, 13781, 13813, 13845, 13877, 13909, 
    13942, 13974, 14006, 14038, 14070, 14102, 14134, 14166, 14199, 14231, 
    14263, 14295, 14327, 14359, 14391, 14423, 14456, 14488, 14520, 14552, 
    14584, 14616, 14648, 14680, 14713, 14745, 14777, 14809, 14841, 14873, 
    14905, 14937, 14970, 15002, 15034, 15066, 15098, 15130, 15162, 15194, 
    15227, 15259, 15291, 15323, 15355, 15387, 15419, 15451, 15484, 15516, 
    15548, 15580, 15612, 15644, 15676, 15708, 15741, 15773, 15805, 15837, 
    15869, 15901, 15933, 15965, 15998, 16030, 16062, 16094, 16126, 16158, 
    16190, 16222, 16255, 16287, 16319, 16351, 16383
], dtype=np.uint16)

def generate_inverse_lut(pwl, output_bits=20):

    max_value = (1 << output_bits) - 1
    lut_size = 1 << EFFECTIVE_BITS
    lut = np.zeros(lut_size, dtype=np.uint32)
    
    for value in range(lut_size):
        idx = np.searchsorted(pwl, value, side="right") - 1
        idx = np.clip(idx, 0, len(pwl) - 2)
        
        y0, y1 = pwl[idx], pwl[idx + 1]
        if y1 == y0:
            x = float(idx)
        else:
            x = idx + (value - y0) / (y1 - y0)
        
        lut[value] = int(round(x * max_value / (len(pwl) - 1)))
    
    return lut


def convert_raw16_to_raw20(input_path, output_path=None):
    inverse_lut = generate_inverse_lut(FORWARD_PWL, OUTPUT_BITS)
    
    with open(input_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint16)
    
    raw_data = raw_data >> (INPUT_BITS - EFFECTIVE_BITS) 
    raw_20bit = inverse_lut[raw_data]
    
    with open(output_path, 'wb') as f:
        f.write(raw_20bit.astype(np.uint32).tobytes())
    
    return output_path

def main():
    if len(sys.argv) < 2:
        print("usage:")
        print(f"  {sys.argv[0]} <raw file name>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_20bit.raw"
    
    try:
        convert_raw16_to_raw20(input_file, output_file)
        print("Done!")
    except Exception as e:
        print(f"fail: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
