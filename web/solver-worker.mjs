import {solve} from './solver.mjs';

// The UI terminates the worker to cancel even in the middle of a linear solve.
self.onmessage = ({data}) => {
  if (data?.type !== 'run') return;
  try {
    self.postMessage(solve(data.vertices, data.faces, data.options, event => self.postMessage(event)));
  } catch (error) {
    self.postMessage({type:'error', message:error.message || String(error)});
  }
};
