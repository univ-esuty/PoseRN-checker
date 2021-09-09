import glob 

def concatnate(dirname, num_joints):
    plyfiles = glob.glob("{}/*.ply".format(dirname))

    num_frames = len(plyfiles)
    num_p = num_frames * num_joints

    output = [
        'ply',
        'format ascii 1.0',
        'element vertex {}'.format(num_p),
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'end_header'
    ]

    for ply in plyfiles:
        with open(ply) as f:
            points = [s.strip() for s in f.readlines()]
            del points[:10]
            output += points

    with open('{}.ply'.format(dirname), mode='w') as f:
        f.write('\n'.join(output))

if __name__ == '__main__':
    # concatnate (path/to/data-root-dir, num_of_joints)
    concatnate('keyframes/opt-mocap', 62) 
    concatnate('keyframes/mv-openpose', 25)

