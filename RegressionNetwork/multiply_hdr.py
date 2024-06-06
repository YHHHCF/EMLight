import util
import os
import numpy as np

base_dir = "./results/semantic_model/results/"
nms = os.listdir(base_dir)

h = util.PanoramaHandler()

for nm in nms:
    if nm.endswith('.exr'):
    	hdr_path = base_dir + nm
    	hdr = h.read_hdr(hdr_path)
    	hdr = hdr * 100
    	util.write_exr(hdr_path, hdr)