import cv2
 
# Read image for contour detection and shape recognition
input_image = cv2.imread("sh3.png")
 
# Make a copy to draw contour outline
input_image_cpy = input_image.copy()


# Convert input image to grayscale
gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

 
# Convert the grayscale image to binary (image binarization opencv python) 
ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('Threshold', binary_img)

# hierarchy variable contains information about the relationship between each contours
#cv2.findContours(src, contour_retrieval, contours_approximation)
#contour_retrieval:
#	cv.RETR_EXTERNAL:retrieves only the extreme outer contours
#	cv.RETR_LIST:retrieves all of the contours without establishing any hierarchical relationships.
#	cv.RETR_TREE:retrieves all of the contours and reconstructs a full hierarchy of nested contours.
#contours_approximation:
#	cv.CHAIN_APPROX_NONE: It will store all the boundary points.
#	cv.CHAIN_APPROX_SIMPLE: It will store number of end points(eg.In case of rectangle it will store 4)

contours_list, hierarchy = cv2.findContours(binary_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # Find contours
print("Jumlah kontur yang terdeteksi = "+str(len(contours_list)-1))

kontur_index = 0


#cv.DrawContours(src, contour, contourIndex, colour, thickness)
#jika contur index melebihi yang terdeteksi maka akan error
contour1 = cv2.drawContours(input_image_cpy, contours_list, kontur_index, (0, 255, 255), 3)

#contour1 = cv2.drawContours(input_image_cpy, contours_list, 2, (0, 255, 255), 3)

#contour1 = cv2.drawContours(input_image_cpy, contours_list, 3, (0, 255, 255), 3)

#contour1 = cv2.drawContours(input_image_cpy, contours_list, 4, (0, 255, 255), 3)

#contour1 = cv2.drawContours(input_image_cpy, contours_list, 0, (0, 255, 255), 3)
	
#JUMLAH TITIK POJOKAN
end_points = cv2.approxPolyDP(contours_list[kontur_index], 0.01 * cv2.arcLength(contours_list[kontur_index], True), True)
print("Jumlah titik tepi kontur ["+str(kontur_index)+"]="+str(len(end_points)))


cv2.imshow('Kontur', contour1)
cv2.waitKey(0)
cv2.destroyAllWindows()