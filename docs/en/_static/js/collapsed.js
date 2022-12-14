var collapsedSections = ['Migration Guides', 'Dataset Zoo', 'Model Zoo', 'Notes', 'API Reference']

$(document).ready(function () {
    $('.model-summary').DataTable({
        "stateSave": false,
        "lengthChange": false,
        "pageLength": 20,
        "order": []
    });
});
