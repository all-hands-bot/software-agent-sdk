import { createServer, Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { ConversationClient } from '../client/conversation-client';

describe('conversation create response loss', () => {
  let server: Server;
  let host: string;
  let creates: number;
  let exists: boolean;
  let reject: boolean;
  let saveBeforeDrop: boolean;
  let reads: number;
  beforeEach(async () => {
    creates = 0;
    exists = false;
    reject = false;
    saveBeforeDrop = true;
    reads = 0;
    server = createServer((req, res) => {
      if (req.method === 'POST') {
        creates++;
        req.resume();
        req.on('end', () => {
          if (reject) {
            res.writeHead(400, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ detail: 'invalid settings' }));
          } else {
            exists = saveBeforeDrop;
            req.socket.destroy();
          }
        });
      } else {
        reads++;
        res.writeHead(exists ? 200 : 404, { 'content-type': 'application/json' });
        res.end(JSON.stringify(exists ? { id: 'test-cid' } : { detail: 'not found' }));
      }
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    host = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });
  afterEach(async () => {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });
  it('recovers the existing conversation without repeating its initial message', async () => {
    const result = await new ConversationClient({ host }).createConversation({
      conversation_id: 'test-cid',
      initial_message: { content: 'run once' },
    });
    expect(result.id).toBe('test-cid');
    expect(creates).toBe(1);
  });
  it('does not conceal server validation failures', async () => {
    reject = true;
    await expect(new ConversationClient({ host }).createConversation({
      conversation_id: 'test-cid',
    })).rejects.toMatchObject({ status: 400 });
    expect(creates).toBe(1);
    expect(reads).toBe(0);
  });
  it('preserves the transport error if no conversation was saved', async () => {
    saveBeforeDrop = false;
    await expect(new ConversationClient({ host }).createConversation({
      conversation_id: 'test-cid',
    })).rejects.toThrow('Unknown request error');
    expect(creates).toBe(1);
    expect(reads).toBe(1);
  });
  it('does not guess a conversation id when the caller supplied none', async () => {
    await expect(new ConversationClient({ host }).createConversation({})).rejects.toThrow('Unknown request error');
    expect(creates).toBe(1);
    expect(reads).toBe(0);
  });
});
