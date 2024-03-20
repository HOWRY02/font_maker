import os
import cv2
import yaml
import imutils
import numpy as np


class SHEETtoPNG:
    """Converter class to convert input sample sheet to character PNGs."""
    def convert(self, sheets_dir, characters_dir, cols=8, rows=8):
        """Convert a sheet of sample writing input to a custom directory structure of PNGs.

        Detect all characters in the sheet as a separate contours and convert each to
        a PNG image in a temp/user provided directory.

        Parameters
        ----------
        sheets_dir : str
            Path to the sheet file to be converted.
        characters_dir : str
            Path to directory to save characters in.
        cols : int, default=8
            Number of columns of expected contours. Defaults to 8 based on the default sample.
        rows : int, default=8
            Number of rows of expected contours. Defaults to 10 based on the default sample.
        """
        
        sheet_images = os.listdir(sheets_dir)
        for image_path in sheet_images:
            image = cv2.imread(sheets_dir + '/' + image_path)
            markers_coords = self.detect_markers(image)
            corners_coords = self.detect_corners(markers_coords)
            transformed_image = self.perspective_transform(image, corners_coords)
            transformed_image = imutils.resize(transformed_image, width=1599)
            tile_coords, qr_code_coords = self.get_tile_coordinates(transformed_image, (rows,cols))

            qr_data = self.detect_qr_code(transformed_image, qr_code_coords)
            characters = self.detect_characters(transformed_image, tile_coords)
            self.save_images(characters, characters_dir, qr_data)
            
            # for tile in tile_coords:
            #     cv2.rectangle(transformed_image, tile[0], tile[1], (0, 0, 255), 1)
            # cv2.imwrite('test' + '/' + image_path, transformed_image)
            # cv2.imshow('Square Detection', transformed_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        width = image.shape[1]
        height = image.shape[0]
        img_area = width*height

        ret,thresh = cv2.threshold(gray,120,255,1)
        contours,h = cv2.findContours(thresh,1,2)

        detected_markers = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            area = cv2.contourArea(contour)
            
            if area>(img_area/2000) and area<(img_area/800):
                if len(approx) > 3 and len(approx) < 12:
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        detected_markers.append((cx, cy))

        detected_markers = self.remove_duplicate_coordinates(detected_markers, width, height)

        markers_coords = self.sort_corners([detected_markers[0], detected_markers[1],
                                    detected_markers[-2], detected_markers[-1]])

        return markers_coords


    def remove_duplicate_coordinates(self, coords, width, height, threshold=1e-2):
        unique_coords = []
        for coord in coords:
            is_unique = True
            for ucoord in unique_coords:
                # Calculate the distance between two coordinates
                distance = ((coord[0]/width - ucoord[0]/width) ** 2 + (coord[1]/height - ucoord[1]/height) ** 2) ** 0.5
                if distance <= threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_coords.append(coord)

        return unique_coords


    def sort_corners(self, corners):
        # Find the centroid of the corners
        centroid = np.mean(corners, axis=0)
        
        # Sort the corners based on their angle relative to the centroid
        sorted_corners = sorted(corners, key=lambda x: np.arctan2(x[1] - centroid[1], x[0] - centroid[0]))
        
        # Now the sorted corners should be in the correct order
        return sorted_corners


    def detect_corners(self, markers_coords):
        # Extract the corner coordinates
        top_l, top_r, bottom_r, bottom_l = markers_coords

        width = max(item[0] for item in markers_coords) - min(item[0] for item in markers_coords)
        height = max(item[1] for item in markers_coords) - min(item[1] for item in markers_coords)

        top_left= (int(top_l[0] - width*0.024), int(top_l[1] + height*0.027))
        top_right = (int(top_r[0] + width*0.03), int(top_r[1] + height*0.027))
        bottom_left = (int(bottom_l[0] - width*0.024), int(bottom_l[1] - height*0.035))
        bottom_right = (int(bottom_r[0] + width*0.03), int(bottom_r[1] - height*0.035))

        corners_coords = [top_left, top_right, bottom_right, bottom_left]

        return corners_coords


    def perspective_transform(self, image, corners_coords):
        top_l, top_r, bottom_r, bottom_l = corners_coords

        # Determine width of new image which is the max distance between 
        # (bottom right and bottom left) or (top right and top left) x-coordinates
        width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
        width = max(int(width_A), int(width_B))

        # Determine height of new image which is the max distance between 
        # (top right and bottom right) or (top left and bottom left) y-coordinates
        height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
        height = max(int(height_A), int(height_B))

        # Construct new points to obtain top-down view of image in 
        # top_r, top_l, bottom_l, bottom_r order
        dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                        [0, height - 1]], dtype = "float32")

        # Convert to Numpy format
        corners_coords = np.array(corners_coords, dtype="float32")

        # Find perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners_coords, dimensions)

        # The transformed image
        transformed_image = cv2.warpPerspective(image, matrix, (width, height))

        return transformed_image


    def get_tile_coordinates(self, transformed_image, tile_size):
        width = transformed_image.shape[1]
        height = transformed_image.shape[0]

        # Calculate the number of tiles in each dimension
        num_rows = tile_size[0]
        num_cols = tile_size[1]
        
        # Calculate the width and height of each tile
        tile_width = width // num_cols
        tile_height = int((height // num_rows)*0.83)
        tile_sample = int((height // num_rows)*0.17)
        
        # List to store tile coordinates
        tile_coords = []
        
        # Iterate through rows
        for row in range(num_rows):
            # Iterate through columns
            for col in range(num_cols):
                # Calculate starting and ending coordinates for the tile
                start_x = col * tile_width
                end_x = start_x + tile_width
                start_y = row * (tile_height + tile_sample) + tile_sample
                end_y = start_y + tile_height
                
                # Append the coordinates of the tile (top_left and bottom right)
                tile_coords.append(((start_x, start_y), (end_x, end_y)))
        
        qr_code_coords = [(tile_coords[6][0][0], tile_coords[6][0][1]),
                        (tile_coords[15][1][0], tile_coords[15][1][1])]
        tile_coords = tile_coords[:6] + tile_coords[8:14] + tile_coords[16:]

        return tile_coords, qr_code_coords


    def detect_qr_code(self, image, qr_code_coords):
        qr_image = image[qr_code_coords[0][1]:qr_code_coords[1][1],
                        qr_code_coords[0][0]:qr_code_coords[1][0]]
        # initialize the cv2 QRCode detector
        detector = cv2.QRCodeDetector()
        # detect and decode
        data, vertices_array, binary_qrcode = detector.detectAndDecode(qr_image)
        # if there is a QR code
        # print the data
        if vertices_array is not None:
            pass
            # for qr_code in vertices_array:
            #     cv2.polylines(qr_image, [qr_code.astype(int)], True, (0, 255, 0), 2)
        else:
            print("There was some error")

        return data


    def clean_image(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY)

        # Find non-zero pixel locations
        # black_pixel_locations = np.where(binary_image == 0)

        # Find maximum and minimum locations

        min_col, max_col, min_row, max_row = None, None, None, None
        # if len(black_pixel_locations[0]) != 0:
        #     max_row, max_col = np.max(black_pixel_locations[0]), np.max(black_pixel_locations[1])
        #     min_row, min_col = np.min(black_pixel_locations[0]), np.min(black_pixel_locations[1])
            # min_col, max_col = np.min(black_pixel_locations[1]), np.max(black_pixel_locations[1])

        result_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        return result_img, min_col, max_col, min_row, max_row


    def detect_characters(self, image, tile_coords):
        w, h = tile_coords[0][1][0] - tile_coords[0][0][0], tile_coords[0][1][1] - tile_coords[0][0][1]
        space_h, space_w = 7 * h // 16, 7 * w // 16

        # Since amongst all the contours, the expected case is that the 4 sided contours
        # containing the characters should have the maximum area, so we loop through the first
        # rows*colums contours and add them to final list after cropping.
        characters = []
        for tile_coord in tile_coords:
            w, h = tile_coord[1][0] - tile_coord[0][0], tile_coord[1][1] - tile_coord[0][1]
            cx, cy = tile_coord[0][0] + w // 2, tile_coord[0][1] + h // 2

            roi = image[cy - space_h : cy + int(space_h*0.9), cx - space_w : cx + space_w]
            cleaned_roi, min_x, max_x, min_y, max_y = self.clean_image(roi)
            if min_x is not None:
                real_roi = cleaned_roi[ :max_y , min_x:max_x]
            else:
                real_roi = cleaned_roi
            characters.append([real_roi, cx, cy])

        return characters


    def save_images(self, characters, characters_dir, qr_data):
        with open('config/glyphs.yaml') as yaml_file:
            glyphs = yaml.safe_load(yaml_file)

        os.makedirs(characters_dir, exist_ok=True)

        # Create directory for each character and save the png for the characters
        # Structure (single sheet): UserProvidedDir/ord(character)/ord(character).png
        # Structure (multiple sheets): UserProvidedDir/sheet_filename/ord(character)/ord(character).png
        for k, images in enumerate(characters):
            if k<len(glyphs[qr_data]):
                character = os.path.join(characters_dir, str(ord(glyphs[qr_data][k])))
                if not os.path.exists(character):
                    os.mkdir(character)
                cv2.imwrite(
                    os.path.join(character, str(ord(glyphs[qr_data][k])) + ".png"),
                    images[0],
                )


if __name__ == "__main__":
    sheets_dir = "sheets/Phuc"
    characters_dir = "characters/Phuc"

    SHEETtoPNG().convert(sheets_dir, characters_dir, cols=8, rows=8)
