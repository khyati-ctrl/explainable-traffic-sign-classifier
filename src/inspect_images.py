import cv2

#loading the image
image = cv2.imread("explainable-traffic-sign-classifier/data/samples/sign.jpg")

if image is None:
    print("Image not found!")
else:
    print("Image loaded successfully")
    print("Image dimensions:", image.shape)

    cv2.imshow("Traffic Sign", image) #displaying the image
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print(image.shape)
    print(image.dtype)
    print(image[0, 0])
    resized = cv2.resize(image, (224, 224)) #resizing the image to 224x224 pixels
    print("Original shape:", image.shape)
    print("Resized shape:", resized.shape)
    cropped = image[100:300, 200:400] #cropping the image to a specific region
    print("Cropped shape:", cropped.shape)
'''    
image.shape- Height, width, and channels
image.dtype- Data type used to store pixel values
image[0, 0]- Pixel values at the top-left corner   
'''
    