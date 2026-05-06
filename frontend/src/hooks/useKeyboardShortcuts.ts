import { useEffect } from "react";

interface KeyboardShortcutHandlers {
  onSubmit: () => void;
  onClear: () => void;
  onCloseSidebar: () => void;
}

export const useKeyboardShortcuts = ({
  onSubmit,
  onClear,
  onCloseSidebar,
}: KeyboardShortcutHandlers): void => {
  useEffect(() => {
    const isMac = navigator.platform.toLowerCase().includes("mac");

    const listener = (event: KeyboardEvent): void => {
      const target = event.target as HTMLElement | null;
      const isTextareaFocused = target?.tagName.toLowerCase() === "textarea";

      if (event.key === "Escape") {
        onCloseSidebar();
      }

      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "l") {
        event.preventDefault();
        onClear();
        return;
      }

      if (!isTextareaFocused) {
        return;
      }

      const shouldSubmitOnEnter = (!isMac && event.key === "Enter" && !event.shiftKey) ||
        (isMac && event.key === "Enter" && event.ctrlKey);

      if (shouldSubmitOnEnter) {
        event.preventDefault();
        onSubmit();
      }
    };

    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [onClear, onCloseSidebar, onSubmit]);
};
