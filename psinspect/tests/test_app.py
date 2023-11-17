import unittest

from psinspect import App


class SOPsinspectTests(unittest.TestCase):
    def test_bunch(self):
        from psinspect.app import Bunch

        b = Bunch(foo="bar")
        self.assertEqual(b.foo, "bar")
        with self.assertRaises(AttributeError):
            b.bar
        b.update(name="John Doe")
        self.assertEqual(b.name, "John Doe")

    def test_app(self):
        my_app = App()
        my_app.initialize()
        my_app.run()
