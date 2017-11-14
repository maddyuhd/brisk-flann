'''
To Do:
    -- Features
        - Replace
        - Duplicate
    -- Threading
        - add.py
        - construct.py
    -- When 2 Construct ?
    -- Final check - better method?
    -- Tweak parameter
    -- Bug - del tree[node]
    -- Deprecate Image input
    -- Implement Test frame-work

Change Log:
-- V2.0.0
    # -- Redis implement
    # -- Handle multiple users (add.py)
    --- added test framework
    ---handle disable and delete (with redis)
    --- CleanUp
        -- temp files
        --search.py
        --searcher.py
        --search2.py
    --- Indexer Class
    --- Hdf5 implement
    --- Nodex index fixed(ll process)
-- V1.2.0
    --- Logging
    --- cleanup clear.py
-- V1.1.2
    --- search2.py with app (feature points)
    --- cleanup tf.cluser
'''

inlocal = True  # False

n_clusters = 8
max_size_lev = 500

logPath = "/var/www/html/system/storage/logs/engine.log"
# logPath = "/home/smacar/Desktop/work/brisk-flann/log/log.log"

success = "0 "
failed = "1 "
