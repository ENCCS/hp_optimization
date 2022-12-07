Hyper Parameter Search using Optuna
===================================

Designing Neural Network learning algorithm requires 
setting many hyper parameters. In this 2 hour workshop 
we will see how we can use the Optuna python package 
to automate the labourious task of finding good hyper 
parameters.


.. prereq::

   This workshop assumes good knowledge of Python and familiarity 
   with designing and training neural networks.


The workshop is organized around introductory material giving 
some background to what hyper parameter optimization is about 
and concepts underlying Bayesian Optimization. The first practical 
part uses a Jupyter Notebook to illustrate how the Optuna package 
works, and highlights that it allows us to essentially optimize any 
black box.

The second practical shows how we can add hyper parameter optimization 
to a simple neural network trained using cross-validation to automate the search.

.. toctree::
   :maxdepth: 1
   :caption: Hyper Parameter Optimization
   
   hp_introduction
   

.. toctree::
   :maxdepth: 1
   :caption: Notebook
   
   notebooks/hyper_parameter_optimization.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Example with cross validation

   cross_validation_example



.. _learner-personas:

Who is the course for?
----------------------

You have experience with training neural networks, but find it 
frustrating to manually tune yout hyper parameters.

You want to learn a tool which can be helpful for optimizing any 
black box, including being useful for design of experiments.


About the course
----------------

This course is developed as part of the EuroCC 
National Competence Center Sweden (ENCCS) training materials.




Credits
-------

The lesson file structure and browsing layout is inspired by and derived from
`work <https://github.com/coderefinery/sphinx-lesson>`_ by `CodeRefinery
<https://coderefinery.org/>`_ licensed under the `MIT license
<http://opensource.org/licenses/mit-license.html>`_. We have copied and adapted
most of their license text.

Instructional Material
^^^^^^^^^^^^^^^^^^^^^^

This instructional material is made available under the
`Creative Commons Attribution license (CC-BY-4.0) <https://creativecommons.org/licenses/by/4.0/>`_.
The following is a human-readable summary of (and not a substitute for) the
`full legal text of the CC-BY-4.0 license
<https://creativecommons.org/licenses/by/4.0/legalcode>`_.
You are free to:

- **share** - copy and redistribute the material in any medium or format
- **adapt** - remix, transform, and build upon the material for any purpose,
  even commercially.

The licensor cannot revoke these freedoms as long as you follow these license terms:

- **Attribution** - You must give appropriate credit (mentioning that your work
  is derived from work that is Copyright (c) ENCCS Hyper Parameter Optimization Workshop and individual contributors and, where practical, linking
  to `<https://enccs.se>`_), provide a `link to the license
  <https://creativecommons.org/licenses/by/4.0/>`_, and indicate if changes were
  made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.
- **No additional restrictions** - You may not apply legal terms or
  technological measures that legally restrict others from doing anything the
  license permits.

With the understanding that:

- You do not have to comply with the license for elements of the material in
  the public domain or where your use is permitted by an applicable exception
  or limitation.
- No warranties are given. The license may not give you all of the permissions
  necessary for your intended use. For example, other rights such as
  publicity, privacy, or moral rights may limit how you use the material.



Software
^^^^^^^^

Except where otherwise noted, the example programs and other software provided
with this repository are made available under the `OSI <http://opensource.org/>`_-approved
`MIT license <https://opensource.org/licenses/mit-license.html>`_.

