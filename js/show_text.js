import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "MetaMuse.ShowText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MuseShowTextNode") {
            function populate(text) {
                if (!text) return;
                const textArr = Array.isArray(text) ? text : [text];

                if (this.widgets) {
                    const pos = this.widgets.findIndex((w) => w.name === "display_text");
                    if (pos !== -1) {
                        for (let i = pos; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = pos;
                    }
                }

                for (const t of textArr) {
                    const w = ComfyWidgets["STRING"](this, "display_text", ["STRING", { multiline: true }], app).widget;
                    if (w && w.inputEl) {
                        w.inputEl.readOnly = true;
                        w.inputEl.style.opacity = "0.85";
                        w.inputEl.style.fontSize = "12px";
                        w.inputEl.style.lineHeight = "1.4";
                    }
                    if (w) {
                        w.value = typeof t === "string" ? t : JSON.stringify(t, null, 2);
                    }
                }

                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (this.size[0] < Math.max(sz[0], 360)) this.size[0] = Math.max(sz[0], 360);
                    if (this.size[1] < Math.max(sz[1], 180)) this.size[1] = Math.max(sz[1], 180);
                    this.setDirtyCanvas(true, true);
                });
            }

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.text) {
                    populate.call(this, message.text);
                }
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                if (this.widgets_values?.length) {
                    populate.call(this, this.widgets_values);
                }
            };
        }
    }
});
