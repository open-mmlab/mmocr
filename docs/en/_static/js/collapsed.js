var collapsedSections = ['Migration Guides', 'API Reference']

$(document).ready(function () {
    $('.model-summary').DataTable({
        "stateSave": false,
        "lengthChange": false,
        "pageLength": 20,
        "order": []
    });
});
