import numpy as np


class ImageBlockGenerator:
    """
    A class for generating masked target patches and a context image with patches removed,
    designed for use in I-JEPA (Image-based Joint Embedding Predictive Architecture).

    The class allows for configurable block sizes, number of target patches,
    and patch dimensions, making it flexible for different image sizes and processing requirements.

    Attributes:
        block_size (int):
            Size of each block in pixels. Determines the granularity of the image division.
        num_target_patches (int):
            Number of target patches to generate from the input image.
        min_target_patch_len (int):
            Minimum length of the target patch in pixels (calculated as blocks * block_size).
        max_target_patch_len (int):
            Maximum length of the target patch in pixels (calculated as blocks * block_size).

    Methods:
        __init__(block_size, num_target_patches, min_target_patch_len, max_target_patch_len):
            Initializes the generator with user-defined parameters.
        generate_blocks(image):
            Generates masked target patches and a context image with the patches removed.
        _validate_image(image):
            Ensures the image dimensions are divisible by the block size.
        _get_random_patch_coordinates(image_height, image_width):
            Generates random coordinates for a patch within the image boundaries.
        _extract_patch(image, top_left, bottom_right):
            Extracts a patch from the image and returns it with metadata.
        _remove_patch_from_context(context_image, top_left, bottom_right):
            Removes a patch from the context image by setting its pixels to zero.
    """

    def __init__(self, block_size=4,
                 num_target_patches=4,
                 min_target_patch_len=2,
                 max_target_patch_len=8):
        """
        Initialize the ImageBlockGenerator with configurable parameters.

        Args:
            block_size (int): Size of each block in pixels.
            num_target_patches (int): Number of target patches to generate.
            min_target_patch_len (int): Minimum target patch length in blocks.
            max_target_patch_len (int): Maximum target patch length in blocks.
        """
        self.block_size = block_size
        self.num_target_patches = num_target_patches
        self.min_target_patch_len = min_target_patch_len * block_size
        self.max_target_patch_len = max_target_patch_len * block_size

    def _validate_image(self, image):
        """
        Validate that the image dimensions are divisible by the block size.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        height, width = image.shape[:2]
        if height % self.block_size != 0 or width % self.block_size != 0:
            raise ValueError(
                "Image dimensions must be divisible by block_size.")

    def _get_random_patch_coordinates(self, image_height, image_width):
        """
        Generate random top-left and bottom-right coordinates for a patch.

        Args:
            image_height (int): Height of the image.
            image_width (int): Width of the image.

        Returns:
            tuple: Top-left and bottom-right coordinates of the patch.
        """
        top_left_y = np.random.choice(np.arange(0, image_height, self.block_size))
        top_left_x = np.random.choice(np.arange(0, image_width, self.block_size))

        patch_x_size = np.random.randint(
            self.min_target_patch_len, self.max_target_patch_len)
        
        patch_y_size = np.random.randint(
            self.min_target_patch_len, self.max_target_patch_len)
        
        bottom_right_y = min(top_left_y + patch_x_size, image_height)
        bottom_right_x = min(top_left_x + patch_y_size, image_width)

        return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

    def _extract_patch(self, image, top_left, bottom_right):
        """
        Extract a patch from the image and return it with the patch's coordinates.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            top_left (tuple): Top-left coordinates (x, y) of the patch.
            bottom_right (tuple): Bottom-right coordinates (x, y) of the patch.

        Returns:
            dict: A dictionary containing the patch and its coordinates.
        """
        top_left_x, top_left_y = top_left
        bottom_right_x, bottom_right_y = bottom_right

        patch = np.zeros_like(image)
        patch[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = \
            image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        return {
            "patch": patch,
            "coords": {
                "top_left_x": top_left_x,
                "top_left_y": top_left_y,
                "bottom_right_x": bottom_right_x,
                "bottom_right_y": bottom_right_y
            }
        }

    def _remove_patch_from_context(self, context_image, top_left, bottom_right):
        """
        Remove a patch from the context image by setting its pixels to zero.

        Args:
            context_image (np.ndarray): Context image as a NumPy array.
            top_left (tuple): Top-left coordinates (x, y) of the patch.
            bottom_right (tuple): Bottom-right coordinates (x, y) of the patch.
        """
        top_left_x, top_left_y = top_left
        bottom_right_x, bottom_right_y = bottom_right
        context_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

    def generate_blocks(self, image):
        """
        Generate masked target patches and a context image with patches removed.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            dict: A dictionary containing the context image and target patches.
        """
        self._validate_image(image)

        context_image = image.copy()
        target_patches = []

        image_height, image_width = image.shape[:2]
        for _ in range(self.num_target_patches):
            top_left, bottom_right = self._get_random_patch_coordinates(
                image_height, image_width)
            patch_data = self._extract_patch(image, top_left, bottom_right)
            target_patches.append(patch_data)
            self._remove_patch_from_context(
                context_image, top_left, bottom_right)

        return {
            "context": context_image,
            "targets": target_patches
        }
