import { createServer, Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { HttpClient } from '../client/http-client';
import { RemoteWorkspace } from '../workspace/remote-workspace';

describe('conversation-scoped requests', () => {
  let server: Server;
  let host: string;
  const urls: string[] = [];
  beforeAll(async () => {
    server = createServer((req, res) => {
      urls.push(req.url!);
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ exit_code: 0, stdout: 'ok', stderr: '' }));
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    host = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });
  afterAll(async () => {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });
  it('scopes workspace commands to their conversation', async () => {
    const workspace = new RemoteWorkspace({
      host,
      workingDir: '/workspace',
      conversationId: 'demo-cid',
    });
    const result = await workspace.executeCommand('pwd');
    expect(result.stdout).toBe('ok');
    expect(urls.pop()).toBe('/api/bash/execute_bash_command?cid=demo-cid');
  });
  it('preserves unscoped behavior and explicit query context', async () => {
    await new HttpClient({ baseUrl: host }).get('/api/file/home');
    expect(urls.pop()).toBe('/api/file/home');
    await new HttpClient({ baseUrl: host, conversationId: 'default' }).get('/api/file/download', {
      params: { cid: 'explicit' },
    });
    expect(urls.pop()).toBe('/api/file/download?cid=explicit');
  });
});
