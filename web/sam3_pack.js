// Puts a download button on the "Pack Assets For Download" node.
//
// ComfyUI has no bulk download, so the pack node writes a zip into the output folder and
// reports its /view URL. Everything below is just surfacing that URL as one click.

import { app } from "../../scripts/app.js";

const NODE = "SAM3PackAssets";

function attach(node, info) {
    if (!info || !info.url) return;
    node._sam3Pack = info;

    if (!node._sam3PackWidget) {
        node._sam3PackWidget = node.addWidget(
            "button",
            "download",
            "download",
            () => {
                const pack = node._sam3Pack;
                if (!pack) return;
                // A plain anchor click keeps the browser's own download UI, which handles
                // large files and resumes far better than anything fetched into memory.
                const link = document.createElement("a");
                link.href = new URL(pack.url, window.location.origin).href;
                link.download = pack.filename || "sam3_assets.zip";
                document.body.appendChild(link);
                link.click();
                link.remove();
            }
        );
        node._sam3PackWidget.serialize = false;
    }
    node._sam3PackWidget.name = info.summary ? `download — ${info.summary}` : "download";
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "sam3.pack.download",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const info = message?.sam3_pack?.[0];
            attach(this, info);
        };

        // Re-running a cached graph does not fire onExecuted, so keep the button across reloads.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            onConfigure?.apply(this, arguments);
            if (data?.sam3_pack) attach(this, data.sam3_pack);
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            onSerialize?.apply(this, arguments);
            if (this._sam3Pack) data.sam3_pack = this._sam3Pack;
        };
    },
});
