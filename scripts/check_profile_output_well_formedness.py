import sys
import json
import argparse

# config option to avoid infinite looping
_max_kernels_per_op = 1024

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", help="ORT profiler output files -- formatted as JSON -- to be checked.",
                        type=str, nargs="+", action='extend')
    return parser.parse_args()

def _check_one_file(input_path: str):
    print(f'Processing file: {input_path}...')
    with open(input_path) as f:
        events = json.load(f)

    last_ts = -1

    for idx, event in enumerate(events):
        if event['ts'] <= last_ts:
            print(f'Timestamps not monotonically increasing at index {idx}:\n{json.dumps(event)}')
            sys.exit(1)
            
        # we don't want any stray kernel events,
        # except for memcpys
        if event['cat'] == 'Kernel' and 'Memcpy' not in event['name']:
            op_name_in_kernel = event['args']['op_name']
            # ensure that we have a "xxxx_kernel_time" event within max_kernels_per_op events
            valid = False
            for i in range(1, _max_kernels_per_op + 1):
                if events[idx - i]['cat'] == 'Kernel':
                    if events[idx - i]['args']['op_name'] != op_name_in_kernel:
                        print(f'Invalid kernel launch found at index {idx}:\n{json.dumps(events[idx - i])}')
                        sys.exit(1)
                    else:
                        continue

                elif (events[idx - i]['cat'] == 'Node' and events[idx - i]['name'].endswith('kernel_time') and
                      events[idx - i]['args']['op_name'] == op_name_in_kernel):
                    valid = True
                    break
                else:
                    print(f'Invalid kernel launch found at index {idx}:\n{json.dumps(events[idx])}')
                    sys.exit(1)
            if not valid:
                print(f'Invalid kernel launch found at index {idx}:\n{json.dumps(events[idx])}')
                sys.exit(1)

    # now check that every kernel_time event with ROCMExecutionProvider has one or more corresponding
    # kernel events
    idx = 0

    while idx < len(events):
        event = events[idx]
        if (event['cat'] == 'Node' and
            event['name'].endswith('kernel_time') and
            'Shape' not in event['name'] and
            'Reshape' not in event['name'] and
            'Unsqueeze' not in event['name'] and
            'Flatten' not in event['name'] and
            (event['args']['provider'] == 'ROCMExecutionProvider' or
             event['args']['provider'] == 'CUDAExecutionProvider')):

            op_name_in_kernel_time_event = event['args']['op_name']
            kernel_time_event_name = event['name']
            valid = False
            for i in range(1, _max_kernels_per_op + 1):
                if events[idx + i]['cat'] == 'Kernel':
                    if (events[idx + i]['args']['op_name'] == op_name_in_kernel_time_event and
                        events[idx + i]['args']['parent_name'] == kernel_time_event_name):
                        valid = True
                        continue
                    else:
                        print(f'Invalid kernel launch found at index {idx + i}:\n{json.dumps(events[idx+i])}')
                        sys.exit(1)
                else:
                    break
            if not valid:
                print(f'Invalid kernel_time event at index {idx}:\n{json.dumps(events[idx])}')
                sys.exit(1)
            else:
                idx += i
        else:
            idx += 1


def _main():
    args = _parse_args()
    for input_file in args.inputs:
        _check_one_file(input_file)

if __name__ == '__main__':
    _main()

