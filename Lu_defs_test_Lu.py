from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]  # Make inline plots larger
import matplotlib as mpl
import matplotlib.cm as mtpltcm

import numpy as np
import cv2
import random

import mediapipe as mp

#espelhado / Mirror / photo

def mirror_1(image):
    mirror=cv2.flip(image,1)
    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title('original')

    plt.subplot(1,2,2)
    plt.imshow(mirror)
    plt.title('Mirroir')

#espelhado / Mirror / video

def mirror(img):

    img2=cv2.flip(img,1)
    return img2

    # aqui em python sem biblioteca 
    # if len(img.shape)==2:
    #     return img[:,::-1]
    # elif len(img.shape)==3:
    #     return img[:,::-1, :]

# side by side
def lado_a_lado_1(imageleft, imageright):
    newimheight = max(imageleft.shape[0], imageright.shape[0])
    newimwidth = imageleft.shape[1] + imageright.shape[1]
    
    newim = np.zeros((newimheight, newimwidth, 3), dtype=np.uint8)

    newim[:imageleft.shape[0], :imageleft.shape[1]] = imageleft
    newim[:imageright.shape[0], -imageright.shape[1]:] = imageright

    return newim

# side by side 
def lado_a_lado_2(imageleft, imageright):
    newimheight = max(imageright.shape[0], imageleft.shape[0])
    newimwidth = imageright.shape[1] + imageleft.shape[1]
    
    newim = np.zeros((newimheight, newimwidth, 3), dtype=np.uint8)

    newim[:imageright.shape[0], :imageright.shape[1]] = imageright
    newim[:imageleft.shape[0], -imageleft.shape[1]:] = imageleft

    return newim

# glow
def glow(img, value = 60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv) # o terceiro, v, é a luminosidade
    
    blurred = cv2.GaussianBlur(hsv[:, :, 2], (49, 49), 0)
    hsv[:, :, 2] = cv2.addWeighted(hsv[:, :, 2], 1, blurred, 1, gamma=0)

    # lim = 255 - value
    # v[v > lim] = 255
    # v[v <= lim] += value

    # final_hsv = cv2.merge((h, s, v))
    # img_1 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR) 
    img_1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    return img_1

# sepia
def sepia(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32)/255
    #solid color
    sepia = np.ones(img.shape)
    sepia[:,:,0] *= 153 #B
    sepia[:,:,1] *= 204 #G
    sepia[:,:,2] *= 255 #R
    #hadamard
    sepia[:,:,0] *= normalized_gray #B
    sepia[:,:,1] *= normalized_gray #G
    sepia[:,:,2] *= normalized_gray #R
    return np.array(sepia, np.uint8)

# blackwhite
def blackwhite(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, blackAndWhiteImage = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage

# x_ray
def x_ray(img):
    
    return 255-img

# cartoon
def cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurImage = cv2.medianBlur(img, 1)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 13)
    color = cv2.bilateralFilter(img, 9, 200, 200)
    cartoon = cv2.bitwise_and(color, color, mask = edges)

    return cartoon

    # img2 = cv2.medianBlur(img, 15)
    # edge = cv2.Canny(img2, 5, 80)
    # kernel = np.ones((3,3), np.uint8)
    # edge = cv2.dilate(edge, kernel, iterations = 1)
    # img2[edge==255] = 0

    # return img2

# draw
def draw(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image into grayscale
    img_invert = cv2.bitwise_not(img_gray) # gray scale to inversion of the image
    img_smoothing = cv2.GaussianBlur(img_invert,(21,21),sigmaX=0,sigmaY=0) # smooting the inverted image

    return cv2.divide(img_gray,255-img_smoothing,scale=256)

# thermal_camera
def thermal_camera(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image into grayscale
    colormap = mpl.cm.jet #initialize the colormap (jet)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=255) #add a normalization
    scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap) #init the mapping
    colors = scalarMap.to_rgba(img_gray)
    return colors

#DividedCam
def myDividedCam(frame, nb=4, carre=True, complete=True):
    '''

    '''
    if carre==True:
        nb_in_width = int(np.ceil(np.sqrt(nb)))
        nb_in_height = int(np.ceil(np.sqrt(nb)))
    else:
        nb_in_height = int(np.floor(np.sqrt(nb)))
        nb_in_width = int(np.ceil(nb/nb_in_height))

    width = int(frame.shape[1]/nb_in_width)
    height = int(frame.shape[0]/nb_in_height)
    dim = (width, height)

    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    newim = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    count=0
    for i in range(1,nb_in_height +1):
        for j in range(1,nb_in_width +1):
            if (complete==False and count < nb) or (complete==True):
                count+=1
                newim[height*(i-1):height*(i), width*(j-1):width*(j)] = resized
    return newim

# ghost_color
def ghost_color(im, frames):
    delay = 30
    frames.append(im)

    im_new = im.copy()
    if len(frames) >= delay:
        im_new[:, :, 0] = frames[-1][:, :, 0]
        im_new[:, :, 1] = frames[int(delay/2)][:, :, 1]
        im_new[:, :, 2] = frames.pop(0)[:, :, 2]

    return im_new

# kaleidoscope
def kaleidoscope(im):
    rows, cols = im.shape[:2]

    tri_corners = np.array([[cols//2, 0], [cols//2, rows//2], [int(cols/2+np.tan(22.5*np.pi/180)*rows/2)-1, 0]], dtype=np.int32)

    mask = np.zeros_like(im[:, :])
    cv2.fillPoly(mask, [tri_corners], (255, 255, 255))

    masked = (mask/255 * im).astype('uint8')
    mirror = masked + cv2.flip(masked, 1)

    ims = [mirror]
    for theta in range(45, 360, 45):
        M = cv2.getRotationMatrix2D((cols/2, rows/2), theta, 1)
        ims.append(cv2.warpAffine(mirror, M, (cols, rows)))#, borderMode=cv2.BORDER_TRANSPARENT))

    return sum(ims)


drawing_utils = mp.solutions.drawing_utils #mp_drawing
drawing_styles = mp.solutions.drawing_styles #mp_drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# for mask filters
def run_filter_with_mediapipe_model(mediapipe_model_getter, mediapipe_based_filter):
    
    cap = cv2.VideoCapture(0)
    
    with mediapipe_model_getter() as model:
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue     # If loading a video, use 'break' instead of 'continue'.

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # print(image)

            results = model.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            result_image = mediapipe_based_filter(image, results)

            cv2.imshow('MediaPipe', result_image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    return image, results

# for mask filters
def draw_holistic_results(image, results, show_hands=False, show_face=True, show_pose=False):
    if show_hands:
        drawing_utils.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
        )

        drawing_utils.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
        )

    if show_face:
        drawing_utils.draw_landmarks(
            image,
            results.face_landmarks,
            mp.solutions.holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(thickness=0, circle_radius=0, color=(255, 255, 255)),
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
        )

    if show_pose:
        drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
        )
    
    return image


# red nose
def red_nose(image, results):
    for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        nose = landmarks[1]  # o ponto do nariz

        # image dimensions
        height, width = image.shape[:2]

        # Center coordinates
        center_coordinates = (int(nose.x*width), int(nose.y*height))
        
        # Radius of circle
        radius = int(-300*nose.z+1) # nunca saberia fazer isso
        
        # Red color in BGR
        color = (0, 0, 255)
        
        # Line thickness of -1 px
        thickness = -1
        
        # Using cv2.circle() method
        # Draw a circle of red color of thickness -1 px
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
        
    return image

# flashing nose
def flashing_nose(image, results):
    for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        nose = landmarks[1]  # o ponto do nariz

        height, width = image.shape[:2]

        center_coordinates = (int(nose.x*width), int(nose.y*height))
        
        radius = int(-300*nose.z+1) # nunca saberia fazer isso
        
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        thickness = -1
        
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
        
    return image


# general
def efect_image_face_1(image, results):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False

    im_height, im_width, _ = image.shape

    obj = cv2.imread("data/groucho-marx_resize.png")

    if results.multi_face_landmarks:
        for face_num, face_landmarks in enumerate(results.multi_face_landmarks):

            nose = face_landmarks.landmark[10]

            scalefactor = (0.15 - 1.15*nose.z)
            
            obj = cv2.resize(obj,None,fx=scalefactor, fy=scalefactor,interpolation = cv2.INTER_LANCZOS4)

            #obj = cv2.resize(obj, dsize=(im_width//4, im_height//4), interpolation = cv2.INTER_LANCZOS4)
            # cv2.imshow('obj', obj)

            mask = 255 * np.ones(obj.shape, obj.dtype)
            
            # center_x, center_y = int(nose.x*im_height), int(nose.y*im_width)+obj.shape[0]//2
            center_x, center_y = int(nose.x*im_width), int(nose.y*im_height)+obj.shape[0]//2
            try:
                image = cv2.seamlessClone(obj, image, mask, (center_x, center_y), cv2.MIXED_CLONE)

            except Exception as e:
                print(e)

    return image

# carnaval
def carnaval(image, results):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False

    im_height, im_width, _ = image.shape

    # obj = cv2.imread("data/groucho-marx_resized.png", cv2.IMREAD_UNCHANGED)
    obj = cv2.imread("data/carnaval.png", cv2.IMREAD_UNCHANGED)
    # obj = cv2.imread("data/cat_with_transparancy.png", cv2.IMREAD_UNCHANGED)


    # pts1 = np.float32([[25, 75],[290, 75],[160, 250]]) # groucho-marx_resized.
    pts1 = np.float32([[204, 142],[511, 142],[354, 300]]) # carnaval
    # pts1 = np.float32([[97, 67],[533, 98],[337, 487]]) # boca
    # pts1 = np.float32([[349, 470],[725, 470],[536, 710]]) # gato


    if results.multi_face_landmarks:
        for face_num, face_landmarks in enumerate(results.multi_face_landmarks):

            eye_1 = face_landmarks.landmark[71] # groucho-marx_resized and carnaval
            eye_2 = face_landmarks.landmark[301]
            nose = face_landmarks.landmark[164]

            # eye_1 = face_landmarks.landmark[39] # boca
            # eye_2 = face_landmarks.landmark[269]
            # nose = face_landmarks.landmark[17]

            # eye_1 = face_landmarks.landmark[54] # gato
            # eye_2 = face_landmarks.landmark[284]
            # nose = face_landmarks.landmark[17]

            pts2 = np.float32([[eye_1.x*im_width, eye_1.y*im_height],[eye_2.x*im_width, eye_2.y*im_height],[nose.x*im_width, nose.y*im_height]])

            M = cv2.getAffineTransform(pts1,pts2)

            dst = cv2.warpAffine(obj,M,(im_width,im_height))


            color = dst[:, :, :3]
            alpha = dst[:, :, 3:] / 255.  # should be an array between 0 and 1

            image = alpha * color + (1-alpha) * image
            image = image.astype(np.uint8)
            # cv2.imshow('dst', dst)

    return image    

# groucho_marx
def groucho_marx(image, results):
    image.flags.writeable = False

    im_height, im_width, _ = image.shape

    obj = cv2.imread("data/groucho-marx_resized.png", cv2.IMREAD_UNCHANGED)

    pts1 = np.float32([[25, 75],[290, 75],[160, 250]]) # groucho-marx_resized.

    if results.multi_face_landmarks:
        for face_num, face_landmarks in enumerate(results.multi_face_landmarks):

            eye_1 = face_landmarks.landmark[71] # groucho-marx_resized and carnaval
            eye_2 = face_landmarks.landmark[301]
            nose = face_landmarks.landmark[164]

            pts2 = np.float32([[eye_1.x*im_width, eye_1.y*im_height],[eye_2.x*im_width, eye_2.y*im_height],[nose.x*im_width, nose.y*im_height]])

            M = cv2.getAffineTransform(pts1,pts2)

            dst = cv2.warpAffine(obj,M,(im_width,im_height))


            color = dst[:, :, :3]
            alpha = dst[:, :, 3:] / 255.  # should be an array between 0 and 1

            image = alpha * color + (1-alpha) * image
            image = image.astype(np.uint8)
            # cv2.imshow('dst', dst)

    return image    

# lips
def lips(image, results):

    image.flags.writeable = False

    im_height, im_width, _ = image.shape

    obj = cv2.imread("data/lips.png", cv2.IMREAD_UNCHANGED)

    pts1 = np.float32([[97, 67],[533, 98],[337, 487]]) # boca


    if results.multi_face_landmarks:
        for face_num, face_landmarks in enumerate(results.multi_face_landmarks):

            eye_1 = face_landmarks.landmark[39] # boca
            eye_2 = face_landmarks.landmark[269]
            nose = face_landmarks.landmark[17]

            pts2 = np.float32([[eye_1.x*im_width, eye_1.y*im_height],[eye_2.x*im_width, eye_2.y*im_height],[nose.x*im_width, nose.y*im_height]])

            M = cv2.getAffineTransform(pts1,pts2)

            dst = cv2.warpAffine(obj,M,(im_width,im_height))


            color = dst[:, :, :3]
            alpha = dst[:, :, 3:] / 255.  # should be an array between 0 and 1

            image = alpha * color + (1-alpha) * image
            image = image.astype(np.uint8)
            # cv2.imshow('dst', dst)

    return image    

#cat rouge
def cat(image, results):

    image.flags.writeable = False

    im_height, im_width, _ = image.shape

    obj = cv2.imread("data/cat_with_transparancy.png", cv2.IMREAD_UNCHANGED)

    pts1 = np.float32([[349, 470],[725, 470],[536, 710]]) # gato

    if results.multi_face_landmarks:
        for face_num, face_landmarks in enumerate(results.multi_face_landmarks):

            eye_1 = face_landmarks.landmark[54] # gato
            eye_2 = face_landmarks.landmark[284]
            nose = face_landmarks.landmark[17]

            pts2 = np.float32([[eye_1.x*im_width, eye_1.y*im_height],[eye_2.x*im_width, eye_2.y*im_height],[nose.x*im_width, nose.y*im_height]])

            M = cv2.getAffineTransform(pts1,pts2)

            dst = cv2.warpAffine(obj,M,(im_width,im_height))


            color = dst[:, :, :3]
            alpha = dst[:, :, 3:] / 255.  # should be an array between 0 and 1

            image = alpha * color + (1-alpha) * image
            image = image.astype(np.uint8)
            # cv2.imshow('dst', dst)

    return image    

# get model
def get_face_mesh_model():
    return mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) 

Holistic = mp.solutions.holistic.Holistic

holistic_model = Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

filters = [mirror, glow, sepia, blackwhite, x_ray, cartoon, draw, thermal_camera, ghost_color, kaleidoscope, carnaval, red_nose, groucho_marx, flashing_nose, lips, cat]

# for filter in filters:
def filter_choise(filter):

    if filter in [carnaval, red_nose, groucho_marx, flashing_nose, lips, cat]:

        last_image, last_results = run_filter_with_mediapipe_model(
            mediapipe_model_getter=get_face_mesh_model,
            mediapipe_based_filter=filter
        )

    else:
        vid = cv2.VideoCapture(0)

        if filter == mirror:

            while(True):
                
                ret, frame = vid.read()

                frame_filter_mirror=cv2.flip(frame,1)
                frame_filter = filter(frame_filter_mirror)
                frame_fin = lado_a_lado_1(frame_filter_mirror, frame_filter)
                
                cv2.imshow('frame', frame_fin)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            vid.release()
            cv2.destroyAllWindows()

        elif filter == ghost_color:
            
            frames = []

            while(True):
            
                ret, frame = vid.read()

                frame_filter_mirror=cv2.flip(frame,1)
                frame_filter = filter(frame_filter_mirror, frames)
            
                cv2.imshow('frame', frame_filter)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            vid.release()
            cv2.destroyAllWindows()
        
        else:
            while(True):
                
                ret, frame = vid.read()

                frame_filter_mirror=cv2.flip(frame,1)
                frame_filter = filter(frame_filter_mirror)
            
                cv2.imshow('frame', frame_filter)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            vid.release()
            cv2.destroyAllWindows()