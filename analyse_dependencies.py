import matlab.engine 

# Start MATLAB engine
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath(eng.pwd()))

# Run the dependency analysis - does not work as hoped
files, products = eng.matlab.codetools.requiredFilesAndProducts('.\cora_matlab\contSet\@interval\plus.m', nargout=2)

# Convert the MATLAB cell array to Python list
files = list(files)

# Print the dependencies
print("Required Files:")
for f in files:
    print(f)

# Stop the engine
eng.quit()
