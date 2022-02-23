import numpy as np
import cv2
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


# This function will compute the exact location in x,y coordinates for the pixel given it's indices p1 and p2
def compute_pixel_locs(h,w,p1):
    p1x = p1/w
    p1y = p1%w
    return int(p1x), int(p1y)


# This function will compute the matrixes for the weights between two nodes of the graph and the degree matrix of the graph
def compute_graph(img):
    # Converting the image to grayscale for processing 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h,w = img.shape[0], img.shape[1]

    print("img shape: ", img.shape)

    # Number of nodes in the graph
    n = h*w
    # Local neighborhood to be considered while defining the neighbours of the graph 
    r = int(min(h,w) / 4) 
    print("n: ", n, "r: ", r)

    # Initializing the weight and the Degree matrix
    W = np.zeros((n, n)) 
    D = np.zeros((n, n))

    sigma_i = 100
    sigma_x = 30

    # Efficient way to create the graph, moving in only the region which are inside a radius distance to the given pixel 
    for x in range(h):
        for y in range(w):
            for u in range(max(0, x-r), min(h, x+r)):
                for v in range(max(0, y-r), min(w, y+r)):
                    rad = np.sqrt((x-u)*(x-u) + (y-v)*(y-v)) 
                
                    overall_weight = 0
                    if (rad < r):
                        magn = np.exp(-((img[x,y] - img[u,v])**2)/ (sigma_i**2))
                        weight = np.exp(-(rad**2)/ (sigma_x**2))
                        overall_weight = magn * weight 

                    # Index of the first point in the single array of pixels
                    p1_index = x*w + y
                    # Index of the second point in the single array of pixels 
                    p2_index = u*w + v

                    W[p1_index, p2_index] = overall_weight  
                    # print("overall weight: ", overall_weight)

    # Initializing the degree mat
    for id in range(0, n):
        D[id,id] = np.sum(W[id,:]) 

    W_img = (W.copy() / W.max()) * 255.0
    W_img = cv2.resize(W_img, (512, 512))
    cv2.imwrite('./figures/graph.png', W_img)    

    return W,  D


# Computing the eigen vectors and eigen values by solving the eigen problem with sparse entries 
def compute_evecs(W, D):
    D_sqrt = np.zeros(D.shape)
    n = D.shape[0]

    for id in range(n): 
        d_root = np.sqrt(D[id,id])
        D_sqrt[id, id] = 1/d_root 

    # Matrix A in the standard form for eigenvector estimation 
    A = np.dot(np.dot(D_sqrt, D - W), D_sqrt)

    evals, evecs = eigsh(A, k=5, which='SM')
    return evals, evecs

# Overal in a smooth fashion 
def overlay_smmoth(img, mask):
    h,w = img.shape[0], img.shape[1]
    c1 = [210, 0, 0]
    c2 = [0, 0, 210]

    for x in range(h):
        for y in range(w):
            al = mask[x,y]
            r = al*c1[0] + (1-al)*c2[0] 
            g = al*c1[1] + (1-al)*c2[1] 
            b = al*c1[2] + (1-al)*c2[2]
            img[x,y,:] = [r,g,b]

    return img  



# This function will overall the translucent mask to the image using the given two segmentation mask
def overlay_mask(img, mask1, mask2):
    h,w = img.shape[0], img.shape[1]
    c1 = [210,0,0]
    c2 = [0,0,210]

    for x in range(h):
        for y in range(w):
            if (mask1[x,y] == 1):
                img[x, y, 0] = 0*img[x, y, 0] + c1[0]
                img[x, y, 1] = 0*img[x, y, 1] + c1[1]
                img[x, y, 2] = 0*img[x, y, 2] + c1[2]
            if (mask2[x,y] == 1):
                img[x, y, 0] = 0*img[x, y, 0] + c2[0]
                img[x, y, 1] = 0*img[x, y, 1] + c2[1]
                img[x, y, 2] = 0*img[x, y, 2] + c2[2]

    return img
    

# This is the main function will perform the ncut algorithm for the image
def ncut(img_path, save_name):
    img = cv2.imread(img_path)
    h_orig,w_orig = img.shape[0], img.shape[1]  
    size = 50
    h = size
    w = int(size * w_orig / h_orig)
    n = h*w

    img = cv2.resize(img, (w, h))
    # img = cv2.GaussianBlur(img, (5,5), 3) # Applying a gaussian to per-process the image 

    W, D = compute_graph(img) 
    print("Graph computation complete!")
    evals, evecs = compute_evecs(W, D)
    print("Eigen vector computation complete!")

    print("Eigen values: ", evals)
    print("Evecs shape: ", evecs.shape)

    # Taking the eigen vector corresponding to the second smallest eigen value 
    dst_src = './figures/results/segmented_' + save_name
    for ev in range(1,5):
        x_sol = evecs[:,ev]

        x_normed = (x_sol - x_sol.min()) / (x_sol.max() - x_sol.min())

        plt.hist(x_sol, bins = 100)
        plt.savefig('./figures/results/hist_' + save_name + str(ev) + '.png') 
        plt.clf()

        th = 0.0

        # Creating a segmask for the obtained cuts 
        cut1_mask = np.zeros((h, w))
        cut2_mask = np.zeros((h, w)) 

        # cut smooth
        cut_smooth = np.zeros((h,w))

        for id in range(n):
            label = x_sol[id]
            x,y = compute_pixel_locs(h,w,id)

            # Normed image 
            cut_smooth[x, y] = x_normed[id]
            if (label > th): 
                cut1_mask[x, y] = 1
            else:
                cut2_mask[x, y] = 1


        th_img = overlay_mask(img.copy(), cut1_mask, cut2_mask)
        seg_img = overlay_smmoth(img.copy(), cut_smooth)
        result_img = np.hstack([img, seg_img, th_img])

        dst_name = dst_src + str(ev) + '.png'
        cv2.imwrite(dst_name, result_img)  


def run_main():
    img_names = ['0_orig_.jpg', '2_orig_.jpg', '5_orig_.jpg', '7_orig_.jpg', '8_orig_.jpg', '9_orig_.jpg', 'shape1.jpeg', 'shape2.jpeg', 'sofa.jpeg', 'sofa2.jpeg']
    img_paths = ['./figures/results/' + im_name for im_name in img_names]

    for id in range(len(img_paths)):
        img_path = img_paths[id]
        img_name = img_names[id][:-4]
        import time
        start_time = time.time()
        # print("time now: ", time.time())
        ncut(img_path, img_name)
        print("time after: ", time.time() - start_time) 


if __name__ == "__main__":
    run_main()

