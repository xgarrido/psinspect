=========
psinspect
=========
.. inclusion-marker-do-not-remove

``psinspect`` is a visual application to check for CMB power spectra and covariance matrices
calculations through `PSpipe <https://github.com/simonsobs/PSpipe>`_.

..
   .. image:: https://img.shields.io/pypi/v/pspy.svg?style=flat
      :target: https://pypi.python.org/pypi/pspy/
.. image:: https://img.shields.io/badge/license-BSD-yellow
   :target: https://github.com/xgarrido/psinspect/blob/master/LICENSE
..
   .. image:: https://img.shields.io/github/actions/workflow/status/simonsobs/pspy/testing.yml?branch=master
      :target: https://github.com/simonsobs/pspy/actions?query=workflow%3ATesting
   .. image:: https://readthedocs.org/projects/pspy/badge/?version=latest
      :target: https://pspy.readthedocs.io/en/latest/?badge=latest
   .. image:: https://codecov.io/gh/simonsobs/pspy/branch/master/graph/badge.svg?token=HHAJ7NQ5CE
      :target: https://codecov.io/gh/simonsobs/pspy
   .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/simonsobs/pspy/master?filepath=notebooks/%2Findex.ipynb

* Free software: BSD license
..
   * ``pspy`` documentation: https://pspy.readthedocs.io.
   * Scientific documentation: https://pspy.readthedocs.io/en/latest/scientific_doc.pdf


..
   Installing
   ----------

   .. code:: shell

       $ pip install pspy [--user]

   You can test your installation by running

   .. code:: shell

       $ test-pspy

   If everything goes fine, no errors will occur. Otherwise, you can report your problem on the `Issues tracker <https://github.com/simonsobs/pspy/issues>`_.

   If you plan to develop ``pspy``, it is better to checkout the latest version by doing

   .. code:: shell

       $ git clone https://github.com/simonsobs/pspy.git /where/to/clone

   Then you can install the ``pspy`` library and its dependencies *via*

   .. code:: shell

       $ pip install -e /where/to/clone

   The ``-e`` option allow the developer to make changes within the ``pspy`` directory without having
   to reinstall at every changes.


The code is part of `PSpipe <https://github.com/simonsobs/PSpipe>`_ the Simons Observatory power spectrum pipeline.
