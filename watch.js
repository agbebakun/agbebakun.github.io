class Watch {

    constructor(connectionString = null, eventCallback = null, log = null) {
        if (!connectionString) {
            connectionString = `ws://${location.host}/ws`;
        }
        this.connectionString = connectionString;
        this.eventCallback = eventCallback;
        this.log = log;

        // State
        this.reconnectTries = 0;
    }

    connect() {
        if (this.log) this.log('WATCH-WS: Opening ' + this.connectionString);
        this.ws = new WebSocket(this.connectionString);

        this.ws.addEventListener('open', () => {
            if (this.log) this.log('WATCH-WS: Opened');
            this.reconnectTries = 0;
        });

        this.ws.addEventListener('message', (wsEvent) => {
            let eventData;
            try {
                eventData = JSON.parse(wsEvent.data);
            } catch (e) {
                if (this.log) this.log('WATCH-WS: Invalid JSON data received: ' + wsEvent.data);
                return;
            }
            if (this.log) this.log('WATCH-WS-EVENT: ' + wsEvent.data);

            const watchEvent = {
                source: this,
                data: eventData,
            };
            if (this.eventCallback) this.eventCallback(watchEvent);
        });

        this.ws.addEventListener('close', () => {
            if (this.log) this.log('WATCH-WS: Closed');

            // Schedule reconnection with backoff
            this.reconnectTries++;
            let reconnectInterval = (1.2 ** Math.min(this.reconnectTries, 10)) * 10;
            setTimeout(() => {
                this.connect(this.connectionString);
            }, reconnectInterval * 1000);
        });

        this.ws.addEventListener('error', (error) => {
            if (this.log) this.log(`WATCH-WS: Error: ${error.message}`);
        });
    }
}

// Hack to export only if imported as a module (top-level await a regexp divided, otherwise an undefined variable divided followed by a comment)
if(0)typeof await/0//0; export default Watch
