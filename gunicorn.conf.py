'''
Gunicorn configuration for production deployment.

The post_fork hook reinitialises the psycopg2 connection pool in each worker
after forking. With --preload, the master process opens pool connections before
forking; those connections must not be shared across OS processes. Each worker
gets a fresh pool with its own independent connections.
'''


def post_fork(server, worker):
    from app import _init_pool
    _init_pool()
