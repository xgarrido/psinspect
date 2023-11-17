import unittest

from psinspect import App


class SOPsinspectTests(unittest.TestCase):
    def test_app(self):
        my_app = App()
        my_app.initialize()
        my_app.run()
