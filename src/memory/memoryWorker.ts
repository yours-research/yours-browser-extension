import { startMemoryWorkerHost } from '@yours/memory/worker';

void startMemoryWorkerHost({
  dbName: 'yours-extension-memory',
  includeBuiltinBlocklist: true,
});
