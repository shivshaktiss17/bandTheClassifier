#!/usr/bin/python

activate_this = '/var/www/webApp/webApp/venv/bin/activate_this.py'
exec(open(activate_this).read(), dict(__file__=activate_this))


import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/webApp/")

from webApp import app as application
application.secret_key = 'shivshaktiss17'

