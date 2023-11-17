.. image:: https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/user-secret.svg
  :target: https://github.com/xgarrido/psinspect
  :width: 110
  :height: 110
  :align: left 

``psinspect`` is a visual application to check for CMB power spectra and covariance matrices
calculations through `PSpipe <https://github.com/simonsobs/PSpipe>`_.

.. image:: https://img.shields.io/pypi/v/psinspect.svg?style=flat
   :target: https://pypi.python.org/pypi/psinspect/
.. image:: https://img.shields.io/github/actions/workflow/status/xgarrido/psinspect/testing.yml?branch=main
   :target: https://github.com/xgarrido/psinspect/actions?query=workflow%3ATesting
.. image:: https://codecov.io/gh/xgarrido/psinspect/branch/main/graph/badge.svg?token=HHAJ7NQ5CE
   :target: https://codecov.io/gh/xgarrido/psinspect
.. image:: https://img.shields.io/badge/license-BSD-yellow
   :target: https://github.com/xgarrido/psinspect/blob/master/LICENSE

..
   .. image:: https://readthedocs.org/projects/pspy/badge/?version=latest
      :target: https://pspy.readthedocs.io/en/latest/?badge=latest
   .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/simonsobs/pspy/master?filepath=notebooks/%2Findex.ipynb

|

Installing
----------

The easiest way is to get the PyPI version with

.. code:: shell

    pip install psinspect [--user]

You can test if everything has been properly installed with

.. code:: shell

    psinspect

If everything goes fine, no errors will occur. Otherwise, you can report your problem on the `Issues tracker <https://github.com/xgarrido/psinspect/issues>`_.

If you plan to develop/change something inside ``psinspect``, it is better to checkout the latest version by doing

.. code:: shell

    git clone https://github.com/xgarrido/psinspect.git /where/to/clone

Then you can install the ``psinspect`` library and its dependencies *via*

.. code:: shell

    pip install -e /where/to/clone

The ``-e`` option allow the developer to make changes within the ``psinspect`` directory without having
to reinstall at every changes.

Using ``psinspect`` at NERSC
----------------------------

At NERSC, after having set ``python`` with ``module load python``, you can follow the same
installation process and everything will be installed in your local home. Then you can go to `NERSC
Jupyter Hub <https://jupyter.nersc.gov>`_ and start a notebook with the following minimal set of
commands

.. code:: python

   from psinspect import App
   my_app = App()
   my_app.initialize()
   my_app.start()

Another (smarter) way is to encapsulate the whole installation stuff within a ``python`` virtual
env. to avoid conflicts with your existing installation. To do so you can follow these command lines

.. code:: shell

   module load python
   python -m venv /where/to/install/your/env
   source /where/to/install/your/env/bin/activate
   python -m pip install ipykernel
   python -m ipykernel install --user --name=psinspect

This will install a new kernel named ``psinspect`` that you can choose when you will create a new
notebook in `NERSC Jupyter Hub <https://jupyter.nersc.gov>`_. Copy-paste the above ``python`` code
and execute the cell.


The code is part of `PSpipe <https://github.com/simonsobs/PSpipe>`_ the Simons Observatory power spectrum pipeline.
