# Create the main directory structure
mkdir -p EventDeblur/basicsr/{archs,data,losses,metrics,models,ops,utils}
mkdir -p EventDeblur/options/{train,test}
mkdir -p EventDeblur/{scripts,tb_logger,experiments,results,docs}

# Create __init__.py files
touch EventDeblur/__init__.py
touch EventDeblur/basicsr/__init__.py
touch EventDeblur/basicsr/archs/__init__.py
touch EventDeblur/basicsr/data/__init__.py
touch EventDeblur/basicsr/losses/__init__.py
touch EventDeblur/basicsr/metrics/__init__.py
touch EventDeblur/basicsr/models/__init__.py
touch EventDeblur/basicsr/ops/__init__.py
touch EventDeblur/basicsr/utils/__init__.py

# Create basic files for the project
