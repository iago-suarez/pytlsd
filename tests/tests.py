import random
import unittest

import cv2
import numpy as np
import pytlsd
from scipy.sparse import csgraph, csr_matrix
from skimage.transform import pyramid_reduce


class StructureDetectionTest(unittest.TestCase):

    def assert_segs_close(self, segs1: np.ndarray, segs2: np.ndarray, tol: float = 0) -> None:
        """
        Checks that two sets of segments are similar.
        """
        if len(segs1) != len(segs2):
            return False

        # Generate segments in both directions
        inverted_segs1 = segs1.reshape(segs1.shape[:-1] + (2, 2))[..., ::-1, :].reshape(segs1.shape)
        inverted_segs2 = segs2.reshape(segs2.shape[:-1] + (2, 2))[..., ::-1, :].reshape(segs2.shape)
        bidir_segs1 = np.vstack([segs1, inverted_segs1])
        bidir_segs2 = np.vstack([segs2, inverted_segs2])

        mat = np.linalg.norm(bidir_segs1[:, None] - bidir_segs2, axis=-1) < tol
        rank = csgraph.structural_rank(csr_matrix(mat))
        self.assertEqual(rank, len(bidir_segs1), f'Assert error: Line segments: '
                                                 f'\n{np.array2string(segs1, precision=2)} and '
                                                 f'\n {np.array2string(segs2, precision=2)} are not equal')

    def test_empty(self) -> None:
        img = np.zeros((100, 100), np.uint8)
        result = pytlsd.lsd(img)
        self.assertEqual(result.shape, (0, 5))

    def test_square(self) -> None:
        img = np.zeros((200, 200), np.uint8)
        x0, x1, y0, y1 = 20, 140, 50, 150
        img[y0:y1, x0:x1] = 255

        result = pytlsd.lsd(img)
        self.assertEqual(result.shape, (4, 5))

        expected = np.array([[x0, y0, x1, y0],
                             [x1, y0, x1, y1],
                             [x1, y1, x0, y1],
                             [x0, y1, x0, y0]])
        self.assert_segs_close(result[:, :4], expected, tol=2.5)

    def test_triangle(self) -> None:
        img = np.zeros((200, 200), np.uint8)
        # Define the triangle
        top = (100, 20)
        left = (50, 160)
        right = (150, 160)
        expected = np.array([[*top, *right], [*right, *left], [*left, *top]])
        # Draw triangle
        cv2.drawContours(img, [expected.reshape(-1, 2)], 0, (255,), thickness=cv2.FILLED)

        result = pytlsd.lsd(img)
        self.assertEqual(result.shape, (3, 5))
        self.assert_segs_close(result[:, :4], expected, tol=2.5)

    def test_batched_triangle(self) -> None:
        img = np.zeros((200, 200), np.uint8)
        # Define the triangle
        top = (100, 20)
        left = (50, 160)
        right = (150, 160)
        expected = np.array([[*top, *right], [*right, *left], [*left, *top]])
        # Draw triangle
        cv2.drawContours(img, [expected.reshape(-1, 2)], 0, (255,), thickness=cv2.FILLED)

        batch_img = [img]
        batch_expected = [expected]
        for _ in range(3):
            batch_img.append(cv2.rotate(batch_img[-1], cv2.ROTATE_90_CLOCKWISE))
            x0, y0, x1, y1 = batch_expected[-1].T
            w = batch_img[-1].shape[1]
            rotated_pts = np.stack([w - y0, x0, w - y1, x1], axis=-1)
            batch_expected.append(rotated_pts)

        batch_img = np.array(batch_img)

        # result = [pytlsd.lsd(b) for b in batch_img]
        result = pytlsd.batched_lsd(batch_img)

        for e, r in zip(batch_expected, result):
            self.assertEqual(r.shape, (3, 5))
            self.assert_segs_close(r[:, :4], e, tol=2.5)

    def test_real_img(self) -> None:
        img = cv2.imread('resources/ai_001_001.frame.0000.color.jpg', cv2.IMREAD_GRAYSCALE)
        segments = pytlsd.lsd(img)
        # Check that it detects at least 500 segments
        self.assertGreater(len(segments), 500)

    def test_batched_real_imgs(self) -> None:
        img = cv2.imread('resources/ai_001_001.frame.0000.color.jpg', cv2.IMREAD_GRAYSCALE)
        batch_size = 8
        # Generate synthetic variations of img
        batch = []
        for i in range(batch_size):
            H = np.eye(3)
            H[0, 2] = 200 * (random.random() - 0.5)
            H[1, 2] = 200 * (random.random() - 0.5)
            s = 1 + 0.5 * random.random()
            rot = (random.random() - 0.5) * 45
            H = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rot, s) @ H
            new_img = cv2.warpAffine(img, H.astype(np.float32), img.shape[::-1])
            batch.append(new_img)

        batch = np.array(batch)

        # Compute the result sequentially
        expected = [pytlsd.lsd(b) for b in batch]

        # Compute the batched result
        result = pytlsd.batched_lsd(batch)

        # Check that the results are the same
        for i in range(batch_size):
            self.assert_segs_close(result[i][:, :4], expected[i][:, :4], tol=0.5)


    def test_with_grads(self) -> None:
        # Read one image
        gray = cv2.imread('resources/ai_001_001.frame.0000.color.jpg', cv2.IMREAD_GRAYSCALE)
        flt_img = gray.astype(np.float64)

        scale_down = 0.8
        resized_img = pyramid_reduce(flt_img, 1 / scale_down, 0.6)
        # Compute gradients
        gx = 0.3 * cv2.Sobel(resized_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        gy = 0.3 * cv2.Sobel(resized_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        gradnorm = 0.5 * np.sqrt(gx ** 2 + gy ** 2)
        gradangle = np.arctan2(gx, -gy)
        # Threshold them
        threshold = 5.
        NOTDEF = -1024.0
        gradangle[gradnorm <= threshold] = NOTDEF

        segments = pytlsd.lsd(resized_img, 1.0, gradnorm=gradnorm, gradangle=gradangle)
        # Check that it detects at least 500 segments
        self.assertGreater(len(segments), 500)
