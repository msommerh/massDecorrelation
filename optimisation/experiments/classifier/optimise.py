# Basic import(s)
import os

# Project import(s)
import adversarial
from adversarial.utils import parse_args
from optimisation.common import *
from run.adversarial import train

# Global variables
PROJECTDIR='/'.join(os.path.realpath(adversarial.__file__).split('/')[:-2] + [''])

# Main function, called by the Spearmint optimisation procedure
def main(job_id, params):

    # Logging
    print "Call to main function (#{})".format(job_id)
    print "  Parameters: {}".format(params)

    # Create temporary patch file
    jobname  = 'patch.{:08d}'.format(job_id)
    filename = os.path.realpath('patches/{}.json'.format(jobname))
    patch    = create_patch(params)
    save_patch(patch, filename)
    
    # Set arguments
    args = parse_args(['--optimise-classifier',
                       '--patch',   filename,
                       '--jobname', 'classifier-' + jobname,
                       '--gpu',
                       '--devices', '3',
                       '--folds',   '3',
                       '--tensorboard'],
                      adversarial=True)

    # Call main script (in the correct directory)
    with cd(PROJECTDIR):
        result = train.main(args)
        pass

    # Ensure correct type, otherwise Spearmint does not accept value
    result = float(result)
    
    return result
